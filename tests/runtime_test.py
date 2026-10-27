"""CPU contracts for resource relocation and backend selection."""

import ast
import hashlib
import os
from pathlib import Path
import pickle
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/source_tree.py"))["load"]()

from _spark.recipes.qwen3_8 import TargetAndDraftOverrides  # noqa: E402
from _spark.kernels.csf import (  # noqa: E402
    CsfProvider,
    register_csf_provider,
    resolve_csf_provider,
)
from _spark.runtime import (  # noqa: E402
    artifact,
    native_manifest,
    recipe_helpers,
    resources_root,
    source_root,
    vendor_root,
    prepare_engine_args,
    configure_engine,
)
from _spark import runtime  # noqa: E402
from _spark.recipes import register_model, resolve_model  # noqa: E402


class Config:
    def __init__(self):
        self.text_config = SimpleNamespace(num_attention_heads=20, num_hidden_layers=48)
        self.text_config.update = lambda fields: vars(self.text_config).update(fields)


class RuntimeTests(unittest.TestCase):
    def test_bootstrap_import_is_cpu_only(self):
        from _spark.runtime import bootstrap

        self.assertTrue(callable(bootstrap.initialize))
        self.assertNotIn("torch", sys.modules)

    def test_checkout_and_installed_paths(self):
        self.assertEqual(source_root(), ROOT / "src")
        self.assertEqual(resources_root(), ROOT / "assets")
        self.assertEqual(vendor_root(), ROOT / "third_party/python")
        self.assertEqual(recipe_helpers(), ROOT / "src/recipes/qwen3_8/helpers")
        with patch.object(runtime, "source_root", return_value=ROOT / ".build/site-packages/vllm/_spark"):
            self.assertEqual(resources_root(), ROOT / ".build/site-packages/vllm/_spark/assets")
            self.assertEqual(vendor_root(), ROOT / ".build/site-packages/vllm/_spark/vendor")
            self.assertEqual(recipe_helpers(), ROOT / ".build/site-packages/vllm/_spark/recipes/qwen3_8/helpers")

    def test_bootstrap_publishes_portable_paths_without_gpu_imports(self):
        from _spark.runtime import bootstrap

        before = {name for name in ("torch", "vllm", "cutlass", "triton") if name in sys.modules}
        directory = ROOT / ".build/runtime-tests"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            cache = Path(temporary) / "fresh-cache"
            with patch.object(runtime, "cache_root", return_value=cache), \
                    patch.object(sys, "path", list(sys.path)), \
                    patch.object(bootstrap, "_initialized", False), \
                    patch.dict(os.environ, {}, clear=True):
                bootstrap.initialize()
                self.assertTrue(cache.is_dir())
                self.assertIn(str(vendor_root()), sys.path)
                self.assertIn(str(recipe_helpers()), sys.path)
                self.assertEqual(os.environ["MBX_DEPTH_CFG"], str(resources_root() / "recipes/qwen3_8/mtp-depth.json"))
                self.assertEqual(os.environ["MBX_PLE_NVME_DIR"], str(cache / "ple-nvme"))
                for key, value in {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "MBX_PLE_MMAP_MODE": "nvme",
                                   "MBX_PLE_ASYNC": "1", "MBX_PLE_ASYNC_MAX": "32768", "MBX_PLE_NVME_THREADS": "4",
                                   "MBX_PLE_QD": "16", "MBX_PLE_PREFILL_CKPT": "1", "MBX_DEPTH_SYNCFREE": "1",
                                   "VLLM_GDN_DECODE_KERNEL": "cuda", "MBX_LOADER_PREPARE": "0"}.items():
                    self.assertEqual(os.environ[key], value)
                bootstrap.initialize()
        self.assertEqual(before, {name for name in ("torch", "vllm", "cutlass", "triton") if name in sys.modules})

    def test_native_artifacts_are_pinned(self):
        for key, record in native_manifest()["files"].items():
            path, actual = artifact(key)
            self.assertEqual(actual["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["bytes"], path.stat().st_size)

    def test_csf_provider_is_separate_from_callers(self):
        original = resolve_csf_provider()
        self.assertEqual(original["provider"].name, "cubin")
        self.assertTrue(original["cubin"].name.endswith(".cubin"))
        register_csf_provider(CsfProvider("source-fixture", "csf_host"))
        replacement = resolve_csf_provider("source-fixture")
        self.assertNotIn("cubin", replacement)
        self.assertEqual(replacement["provider"].op_name, original["provider"].op_name)
        with self.assertRaises(ValueError):
            resolve_csf_provider("missing")

    def test_original_compression_function_is_preserved(self):
        source = (recipe_helpers() / "mbx_marlin_csf.py").read_text(encoding="utf-8")
        actual = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "compress")
        expected = ast.parse((source_root() / "kernels/csf/original/compress.py").read_text(encoding="utf-8")).body[0]
        self.assertEqual(ast.dump(actual, include_attributes=False), ast.dump(expected, include_attributes=False))

    def test_yarn_keeps_model_fields_and_is_picklable(self):
        overrides = {"text_config": {"max_position_embeddings": 409600, "rope_parameters": {"rope_type": "yarn", "factor": 1.5625}}}
        transform = pickle.loads(pickle.dumps(TargetAndDraftOverrides(overrides)))
        target = Config()
        self.assertIs(transform(target), target)
        self.assertEqual(target.text_config.num_attention_heads, 20)
        self.assertEqual(target.text_config.max_position_embeddings, 409600)
        self.assertEqual(overrides["text_config"]["rope_parameters"]["factor"], 1.5625)

    def test_runtime_helpers_have_no_workspace_paths(self):
        for name in ("mbx_depth.py", "mbx_marlin_csf.py", "native_sampling.py", "mbx_loader.py"):
            source = (recipe_helpers() / name).read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn(str(Path.home()), node.value)
                    self.assertNotIn("research/windows/", node.value)

    def test_qwen_profile_settings_and_order_are_preserved(self):
        from _spark.kernels import cute, csf
        from _spark.runtime import bootstrap

        original_resolve = csf.resolve_csf_provider
        for fast in (False, True):
            calls = []
            options = {"model": "qwen3.8-flash-next", "draft_fp8": fast, "draft_head2": fast}
            args = SimpleNamespace(model="checkpoint", additional_config={"rtx_spark": options}, hf_overrides={
                "text_config": {"max_position_embeddings": 409600}})
            def select(name):
                calls.append(("csf", name))
                return original_resolve(name)
            with patch.object(bootstrap, "initialize", side_effect=lambda: calls.append("initialize")), \
                    patch.object(cute, "configure", side_effect=lambda mode: calls.append(("cute", mode))), \
                    patch.object(csf, "resolve_csf_provider", side_effect=select), \
                    patch.dict(os.environ, {}, clear=True):
                prepare_engine_args(args)
                self.assertEqual(calls, ["initialize", ("cute", "auto"), ("csf", "cubin")])
                self.assertEqual(os.environ["VLLM_DISABLED_KERNELS"], "FlashInferCutedslMxfp8LinearKernel,FlashInferTrtllmMxfp8LinearKernel,DeepGemmMxfp8BmmLinearKernel,HummingMxfp8LinearKernel")
                self.assertEqual(os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"], "1")
                self.assertEqual(os.environ["MBX_MARLIN_CSF"], "1")
                self.assertEqual(os.environ["MBX_MARLIN_CSF_PROVIDER"], "cubin")
                self.assertEqual(os.environ["MBX_DRAFT_FP8"], "1" if fast else "0")
                self.assertEqual("MBX_DRAFT_HEAD2" in os.environ, fast)
                if fast:
                    self.assertEqual(os.environ["MBX_DRAFT_HEAD2"], str(resources_root() / "recipes/qwen3_8/qwen3.8-head2.basis.pt"))
                    self.assertEqual(os.environ["MBX_DRAFT_HEAD2_K"], "2048")
                    self.assertEqual(os.environ["MBX_DRAFT_HEAD2_SCORER"], "fp8")
                self.assertIsInstance(args.hf_overrides, TargetAndDraftOverrides)

    def test_draft_flags_reject_string_booleans(self):
        for key in ("draft_fp8", "draft_head2"):
            args = SimpleNamespace(model="Qwen3.8-Flash-Next-FP4-Local", additional_config={"rtx_spark": {key: "false"}})
            with self.assertRaises(TypeError):
                prepare_engine_args(args)

    def test_another_model_can_register_without_runtime_edits(self):
        calls = []

        def prepare(args, options):
            calls.append(("prepare", options["custom_setting"]))
            args.ready = True

        def validate(config, options):
            calls.append(("validate", options["custom_setting"]))

        register_model("runtime-test-model", prepare, validate)
        arguments = SimpleNamespace(model="another-checkpoint", additional_config={
            "rtx_spark": {"model": "runtime-test-model", "custom_setting": 7}})
        prepare_engine_args(arguments)
        self.assertTrue(arguments.ready)
        self.assertIs(configure_engine(arguments), arguments)
        self.assertEqual(calls, [("prepare", 7), ("validate", 7)])
        with self.assertRaises(ValueError):
            register_model("runtime-test-model", prepare, validate)

    def test_unknown_explicit_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported RTX Spark model recipe"):
            resolve_model(SimpleNamespace(model="anything"), name="not-registered")

    def test_ordinary_vllm_models_do_not_select_a_recipe(self):
        arguments = SimpleNamespace(model="ordinary-model", additional_config={})
        self.assertIsNone(resolve_model(arguments))
        self.assertIsNone(prepare_engine_args(arguments))
        self.assertIs(configure_engine(arguments), arguments)


if __name__ == "__main__":
    unittest.main()
