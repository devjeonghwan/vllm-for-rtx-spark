"""CPU contracts for resource relocation and backend selection."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import pickle
import runpy
import subprocess
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()

from _rtx_spark.recipes.qwen3_8 import TargetAndDraftOverrides  # noqa: E402
from _rtx_spark.recipes import qwen3_8  # noqa: E402
from _rtx_spark.kernels.csf import (  # noqa: E402
    CsfProvider,
    register_csf_provider,
    resolve_csf_provider,
)
from _rtx_spark.runtime import (  # noqa: E402
    artifact,
    native_manifest,
    recipe_helpers,
    resources_root,
    source_root,
    vendor_root,
    prepare_engine_args,
    configure_engine,
)
from _rtx_spark import runtime  # noqa: E402
from _rtx_spark.recipes import register_model, resolve_model  # noqa: E402


class Config:
    def __init__(self):
        self.text_config = SimpleNamespace(num_attention_heads=20, num_hidden_layers=48)
        self.text_config.update = lambda fields: vars(self.text_config).update(fields)


class RuntimeTests(unittest.TestCase):
    def temporary_directory(self):
        parent = ROOT / ".build/runtime-tests"
        parent.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent)

    def prepare_depth(self, minimum, speculative=None):
        from _rtx_spark.kernels import cute, csf
        from _rtx_spark.runtime import bootstrap

        args = SimpleNamespace(
            model="Qwen3.8-Flash-Next-FP4-Local",
            additional_config={"rtx_spark": {"draft_min_depth": minimum}},
            speculative_config=speculative,
        )
        provider = {"library": Path("fixture.dll"), "cubin": Path("fixture.cubin")}
        with patch.object(bootstrap, "initialize"), patch.object(cute, "configure"), \
                patch.object(csf, "resolve_csf_provider", return_value=provider):
            prepare_engine_args(args)
        return Path(os.environ["MBX_DEPTH_CFG"])

    def test_bootstrap_import_is_cpu_only(self):
        from _rtx_spark.runtime import bootstrap

        self.assertTrue(callable(bootstrap.initialize))
        self.assertNotIn("torch", sys.modules)

    def test_checkout_and_installed_paths(self):
        self.assertEqual(source_root(), ROOT / "src")
        self.assertEqual(resources_root(), ROOT / "assets")
        self.assertEqual(vendor_root(), ROOT / "third_party/python")
        self.assertEqual(recipe_helpers(), ROOT / "src/recipes/qwen3_8/helpers")
        installed = ROOT / ".build/site-packages/vllm/_rtx_spark"
        with patch.object(runtime, "source_root", return_value=installed):
            self.assertEqual(resources_root(), ROOT / ".build/site-packages/vllm/_rtx_spark/assets")
            self.assertEqual(vendor_root(), ROOT / ".build/site-packages/vllm/_rtx_spark/vendor")
            self.assertEqual(recipe_helpers(), ROOT / ".build/site-packages/vllm/_rtx_spark/recipes/qwen3_8/helpers")

    def test_bootstrap_publishes_portable_paths_without_gpu_imports(self):
        from _rtx_spark.runtime import bootstrap

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
                self.assertNotIn(str(recipe_helpers()), sys.path)
                for key in ("MBX_DEPTH_CFG", "MBX_PLE_MMAP_MODE", "MBX_PLE_NVME_DIR",
                            "MBX_LOADER_PREPARE", "MBX_DRAFT_FP8"):
                    self.assertNotIn(key, os.environ)
                for key, value in {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_V2_MODEL_RUNNER": "1",
                                   "USE_LIBUV": "0", "VLLM_USE_RUST_FRONTEND": "0"}.items():
                    self.assertEqual(os.environ[key], value)
                bootstrap.initialize()
        self.assertEqual(before, {name for name in ("torch", "vllm", "cutlass", "triton") if name in sys.modules})

    def test_native_artifacts_exist_with_the_required_architecture(self):
        for key, record in native_manifest()["files"].items():
            path, actual = artifact(key)
            self.assertEqual(actual, record)
            self.assertTrue(path.is_file())
            with path.open("rb") as handle:
                header = handle.read(64)
                if key == "csf_cubin":
                    self.assertEqual(header[:4], b"\x7fELF")
                    self.assertEqual(struct.unpack_from("<H", header, 18)[0], 190)
                else:
                    self.assertEqual(header[:2], b"MZ")
                    handle.seek(struct.unpack_from("<I", header, 60)[0])
                    self.assertEqual(handle.read(6), b"PE\0\0\x64\xaa")

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
        from _rtx_spark.kernels import cute, csf
        from _rtx_spark.runtime import bootstrap

        original_resolve = csf.resolve_csf_provider
        for fast in (False, True, None):
            calls = []
            expected_fast = fast is not False
            options = {"model": "qwen3.8-flash-next"}
            if fast is not None:
                options.update(draft_fp8=fast, draft_head2=fast)
            args = SimpleNamespace(model="checkpoint", additional_config={"rtx_spark": options}, hf_overrides={
                "text_config": {"max_position_embeddings": 409600}})
            def select(name):
                calls.append(("csf", name))
                return original_resolve(name)
            with patch.object(bootstrap, "initialize", side_effect=lambda: calls.append("initialize")), \
                    patch.object(runtime, "cache_root", return_value=ROOT / ".build/runtime-tests/profile-cache"), \
                    patch.object(cute, "configure", side_effect=lambda mode: calls.append(("cute", mode))), \
                    patch.object(csf, "resolve_csf_provider", side_effect=select), \
                    patch.dict(os.environ, {}, clear=True):
                prepare_engine_args(args)
                self.assertEqual(calls[0], "initialize")
                self.assertEqual([call for call in calls if call != "initialize"],
                                 [("cute", "auto"), ("csf", "cubin")])
                self.assertLess(max(index for index, call in enumerate(calls) if call == "initialize"),
                                calls.index(("cute", "auto")))
                self.assertEqual(os.environ["MBX_DEPTH_CFG"], str(resources_root() / "recipes/qwen3_8/mtp-depth.json"))
                self.assertEqual(json.loads(Path(os.environ["MBX_DEPTH_CFG"]).read_bytes())["min"], 3)
                for key, value in {"MBX_PLE_MMAP_MODE": "nvme", "MBX_PLE_ASYNC": "1",
                                   "MBX_PLE_ASYNC_MAX": "32768", "MBX_PLE_NVME_THREADS": "4",
                                   "MBX_PLE_QD": "16", "MBX_PLE_PREFILL_CKPT": "1", "MBX_DEPTH_SYNCFREE": "1",
                                   "VLLM_GDN_DECODE_KERNEL": "cuda", "MBX_LOADER_PREPARE": "0"}.items():
                    self.assertEqual(os.environ[key], value)
                self.assertEqual(os.environ["VLLM_DISABLED_KERNELS"], "FlashInferCutedslMxfp8LinearKernel,FlashInferTrtllmMxfp8LinearKernel,DeepGemmMxfp8BmmLinearKernel,HummingMxfp8LinearKernel")
                self.assertEqual(os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"], "1")
                self.assertEqual(os.environ["MBX_MARLIN_CSF"], "1")
                self.assertEqual(os.environ["MBX_MARLIN_CSF_PROVIDER"], "cubin")
                self.assertEqual(os.environ["MBX_DRAFT_FP8"], "1" if expected_fast else "0")
                self.assertEqual("MBX_DRAFT_HEAD2" in os.environ, expected_fast)
                if expected_fast:
                    self.assertEqual(os.environ["MBX_DRAFT_HEAD2"], str(resources_root() / "recipes/qwen3_8/qwen3.8-head2.basis.pt"))
                    self.assertEqual(os.environ["MBX_DRAFT_HEAD2_K"], "2048")
                    self.assertEqual(os.environ["MBX_DRAFT_HEAD2_SCORER"], "fp8")
                self.assertIsInstance(args.hf_overrides, TargetAndDraftOverrides)

    def test_draft_flags_reject_string_booleans(self):
        for key in ("draft_fp8", "draft_head2"):
            args = SimpleNamespace(model="Qwen3.8-Flash-Next-FP4-Local", additional_config={"rtx_spark": {key: "false"}})
            with self.assertRaises(TypeError):
                prepare_engine_args(args)

    def test_draft_min_depth_rejects_invalid_values_before_initialization(self):
        from _rtx_spark.runtime import bootstrap

        for value, error in ((True, TypeError), (False, TypeError), (3.0, TypeError),
                             ("3", TypeError), (None, TypeError), (0, ValueError), (-1, ValueError)):
            with self.subTest(value=value), patch.object(bootstrap, "initialize") as initialize:
                with self.assertRaises(error):
                    prepare_engine_args(SimpleNamespace(
                        model="Qwen3.8-Flash-Next-FP4-Local",
                        additional_config={"rtx_spark": {"draft_min_depth": value}},
                        speculative_config={"num_speculative_tokens": 7},
                    ))
                initialize.assert_not_called()

    def test_draft_min_depth_checks_dict_json_and_model_speculative_limits(self):
        configurations = (
            {"num_speculative_tokens": 7}, {"num-speculative-tokens": 7},
            '{"num_speculative_tokens": 7}', SimpleNamespace(num_speculative_tokens=7),
        )
        with self.temporary_directory() as directory, \
                patch.object(runtime, "cache_root", return_value=Path(directory)), \
                patch.dict(os.environ, {}, clear=True):
            for speculative in configurations:
                with self.subTest(speculative=speculative):
                    self.prepare_depth(7, speculative)
                    with self.assertRaisesRegex(ValueError, "exceeds"):
                        self.prepare_depth(8, speculative)

    def test_draft_min_depth_is_rechecked_against_final_engine_config(self):
        config = SimpleNamespace(model="Qwen3.8-Flash-Next-FP4-Local", additional_config={
            "rtx_spark": {"draft_min_depth": 3}}, speculative_config=SimpleNamespace(num_speculative_tokens=2))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            configure_engine(config)

    def test_draft_min_depth_changes_only_min_and_overrides_external_path(self):
        original_path = resources_root() / "recipes/qwen3_8/mtp-depth.json"
        original_bytes = original_path.read_bytes()
        original_policy = json.loads(original_bytes)
        with self.temporary_directory() as directory, \
                patch.object(runtime, "cache_root", return_value=Path(directory)), \
                patch.dict(os.environ, {"MBX_DEPTH_CFG": "external-policy.json"}, clear=True):
            selected = self.prepare_depth(1, {"num_speculative_tokens": 7})
            expected = dict(original_policy, min=1)
            self.assertEqual(json.loads(selected.read_bytes()), expected)
            self.assertTrue(selected.is_absolute())
            self.assertEqual(selected.parent, Path(directory) / "recipe-configs")
            self.assertEqual(selected.name, "qwen3_8-mtp-" + hashlib.sha256(selected.read_bytes()).hexdigest() + ".json")
            self.assertEqual(original_path.read_bytes(), original_bytes)
            self.assertEqual(original_policy["min"], 3)

    def test_omitted_draft_min_depth_preserves_external_policy_without_cache_writes(self):
        from _rtx_spark.kernels import cute, csf
        from _rtx_spark.runtime import bootstrap

        with self.temporary_directory() as directory, \
                patch.object(runtime, "cache_root", return_value=Path(directory) / "unused-cache"), \
                patch.object(bootstrap, "initialize"), \
                patch.object(cute, "configure"), patch.object(csf, "resolve_csf_provider", return_value={
                    "library": Path("fixture.dll")}), \
                patch.dict(os.environ, {"MBX_DEPTH_CFG": "external-policy.json"}, clear=True):
            args = SimpleNamespace(model="Qwen3.8-Flash-Next-FP4-Local", additional_config={})
            prepare_engine_args(args)
            self.assertEqual(os.environ["MBX_DEPTH_CFG"], "external-policy.json")
            self.assertFalse((Path(directory) / "unused-cache").exists())

    def test_draft_min_depth_policies_are_reused_and_isolated(self):
        with self.temporary_directory() as directory, \
                patch.object(runtime, "cache_root", return_value=Path(directory)), \
                patch.dict(os.environ, {}, clear=True):
            first = self.prepare_depth(3, {"num_speculative_tokens": 7})
            stamp = first.stat().st_mtime_ns
            second = self.prepare_depth(1, {"num_speculative_tokens": 7})
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_bytes())["min"], 3)
            self.assertEqual(json.loads(second.read_bytes())["min"], 1)
            self.assertEqual(self.prepare_depth(3), first)
            self.assertEqual(first.stat().st_mtime_ns, stamp)

    def test_draft_min_depth_concurrent_publication_has_no_partial_files(self):
        with self.temporary_directory() as directory, \
                patch.object(runtime, "cache_root", return_value=Path(directory)), \
                patch.dict(os.environ, {}, clear=True):
            def publish(_):
                qwen3_8._configure_draft_min_depth({"draft_min_depth": 3})
                return json.loads(Path(os.environ["MBX_DEPTH_CFG"]).read_bytes())

            with ThreadPoolExecutor(max_workers=8) as executor:
                policies = list(executor.map(publish, range(24)))
            self.assertTrue(all(policy["min"] == 3 for policy in policies))
            self.assertEqual(len(list((Path(directory) / "recipe-configs").iterdir())), 1)

    def test_changed_cached_policy_is_rejected_without_overwrite(self):
        with self.temporary_directory() as directory, \
                patch.object(runtime, "cache_root", return_value=Path(directory)), \
                patch.dict(os.environ, {}, clear=True):
            selected = self.prepare_depth(3)
            selected.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Cached MTP policy has changed"):
                self.prepare_depth(3)
            self.assertEqual(selected.read_text(encoding="utf-8"), "{}")

    def test_policy_write_failure_does_not_publish_or_select_partial_json(self):
        with self.temporary_directory() as directory, \
                patch.object(runtime, "cache_root", return_value=Path(directory)), \
                patch.dict(os.environ, {"MBX_DEPTH_CFG": "external-policy.json"}, clear=True), \
                patch.object(qwen3_8.os, "fsync", side_effect=OSError("fixture write failure")):
            with self.assertRaisesRegex(OSError, "fixture write failure"):
                self.prepare_depth(3)
            self.assertEqual(os.environ["MBX_DEPTH_CFG"], "external-policy.json")
            self.assertEqual(list((Path(directory) / "recipe-configs").iterdir()), [])

    def test_selected_depth_policy_is_read_by_original_helper_and_inherited_by_child(self):
        source = (recipe_helpers() / "mbx_depth.py").read_text(encoding="utf-8")
        reader = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "_read_fresh")
        module = ast.Module(body=[reader], type_ignores=[])
        with self.temporary_directory() as directory, \
                patch.object(runtime, "cache_root", return_value=Path(directory)), \
                patch.dict(os.environ, {}, clear=True):
            selected = self.prepare_depth(3, {"num_speculative_tokens": 7})
            with ExitStack() as opened:
                namespace = {"CFG": str(selected), "json": json,
                             "open": lambda path: opened.enter_context(open(path))}
                exec(compile(module, "original-mbx-depth-reader", "exec"), namespace)
                self.assertEqual(namespace["_read_fresh"]()["min"], 3)
            child = subprocess.run([sys.executable, "-c", (
                "import json,os; from pathlib import Path; "
                "p=Path(os.environ['MBX_DEPTH_CFG']); "
                "print(json.dumps([str(p),json.loads(p.read_bytes())['min']]))"
            )], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(child.stdout), [str(selected), 3])

    def test_another_model_can_register_without_runtime_edits(self):
        from _rtx_spark.optimizations import Option

        calls = []

        def prepare(args, options):
            calls.append(("prepare", options["custom_setting"]))
            args.ready = True

        def validate(config, options):
            calls.append(("validate", options["custom_setting"]))

        def validate_setting(arguments, value):
            calls.append(("option_validate", value))

        def apply_setting(arguments, value):
            calls.append(("option_apply", value))
            arguments.optimization_value = value

        register_model("runtime-test-model", prepare, validate, options=(
            Option("custom_setting", int, 7, apply=apply_setting, validate=validate_setting),))
        arguments = SimpleNamespace(model="another-checkpoint", additional_config={
            "rtx_spark": {"model": "runtime-test-model"}})
        prepare_engine_args(arguments)
        self.assertTrue(arguments.ready)
        self.assertEqual(arguments.optimization_value, 7)
        self.assertIs(configure_engine(arguments), arguments)
        self.assertEqual(calls, [("option_validate", 7), ("prepare", 7), ("option_apply", 7),
                                 ("option_validate", 7), ("validate", 7)])
        previous = list(calls)
        with self.assertRaises(TypeError):
            prepare_engine_args(SimpleNamespace(model="another-checkpoint", additional_config={
                "rtx_spark": {"model": "runtime-test-model", "custom_setting": "7"}}))
        self.assertEqual(calls, previous)
        with self.assertRaises(ValueError):
            register_model("runtime-test-model", prepare, validate)

    def test_plugin_registers_new_architectures_lazily_without_replacing_vllm_models(self):
        from _rtx_spark import plugin, recipes
        from _rtx_spark.runtime import bootstrap
        from _rtx_spark.runtime.vllm import get_adapter

        adapter = get_adapter()
        calls = []
        supported = {"Qwen4ExpForConditionalGeneration"}

        class Registry:
            @staticmethod
            def get_supported_archs():
                return supported

            @staticmethod
            def register_model(name, implementation):
                calls.append((name, implementation))
                supported.add(name)

        fake_vllm = ModuleType("vllm")
        fake_vllm.ModelRegistry = Registry
        with patch.dict(recipes._MODELS, {}, clear=True), \
                patch.dict(adapter._registered_architectures, {}, clear=True), \
                patch.dict(sys.modules, {"vllm": fake_vllm}), patch.object(bootstrap, "initialize"):
            recipes.register_model(
                "fixture-architecture", lambda args, options: None, lambda config, options: None,
                architectures=("FixtureArchitecture",),
                model_classes={"FixtureArchitecture": "fixture_private.module:LazyModel"})
            plugin.register()
            plugin.register()
            self.assertEqual(calls, [("FixtureArchitecture", "fixture_private.module:LazyModel")])
            self.assertNotIn("fixture_private.module", sys.modules)
            arguments = SimpleNamespace(model="custom-checkpoint", hf_config=SimpleNamespace(
                architectures=["FixtureArchitecture"]))
            self.assertIs(recipes.resolve_model(arguments), recipes.resolve_model(arguments, name="fixture-architecture"))

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
