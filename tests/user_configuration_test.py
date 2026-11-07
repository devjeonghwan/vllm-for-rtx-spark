"""Custom config precedence, immutable policy inputs and real spawn replay."""

from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(ROOT / "scripts/load_source_package.py")["load"]()
from _rtx_spark import runtime
from _rtx_spark.kernels import cute, csf
from _rtx_spark.optimizations import compilation
from _rtx_spark.recipes import qwen3_8_flash_next as qwen
from _rtx_spark.recipes.qwen3_8_flash_next import dflash_nvfp4_core as core
from _rtx_spark.runtime import bootstrap
from _rtx_spark.runtime.depth import get_depth_helper


def _spawn_read(queue, config):
    try:
        bootstrap.initialize()
        compilation.prepare_compilation_policy(config)
        runtime.configure_engine(config)
        options = config.additional_config["rtx_spark"]
        result = {"model": options["model"], "spawn": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")}
        if options["model"] == "none":
            result["depth_disabled"] = get_depth_helper(config) is None
        else:
            settings, _ = qwen._head2_settings(qwen._validate_options(options))
            result["head2"] = settings
            result["environment"] = [os.environ[name] for name in
                ("MBX_DRAFT_HEAD2", "MBX_DRAFT_HEAD2_K", "MBX_DRAFT_HEAD2_SCORER")]
            result["policy"] = json.loads(Path(os.environ["MBX_DEPTH_CFG"]).read_bytes())
        queue.put(result)
    except BaseException as error:
        queue.put({"error": repr(error)})


class UserConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        parent = ROOT / ".build/user-configuration-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=parent)))
        self.stack.enter_context(patch.dict(os.environ))
        for name in list(os.environ):
            if name.startswith(("MBX_DRAFT", "MBX_DEPTH", "MBX_PLE")):
                os.environ.pop(name)
        os.environ.pop("VLLM_USE_BREAKABLE_CUDAGRAPH", None)
        self.stack.enter_context(patch.object(qwen, "_head2_projection", {}))
        self.stack.enter_context(patch.object(qwen, "_depth_projection", None))
        self.stack.enter_context(patch.object(compilation, "_projection", None))
        self.stack.enter_context(patch.object(bootstrap, "_initialized", False))
        os.environ["LOCALAPPDATA"] = str(self.directory / "local")
        self.stack.enter_context(patch.object(runtime, "cache_root", return_value=self.directory / "cache"))
        self.basis = self.directory / "custom.basis.pt"
        self.basis.write_bytes(b"CPU fixture; no tensor loading")

    def prepare(self, **options):
        args = NS(model="custom-variant", speculative_config={"num_speculative_tokens": 7},
                  parallel_config=NS(tensor_parallel_size=1),
                  additional_config={"rtx_spark": dict(model="qwen3.8-flash-next", **options)})
        with patch.object(bootstrap, "initialize"), patch.object(cute, "configure"), \
                patch.object(csf, "resolve_csf_provider", return_value={"library": Path("fixture.dll")}):
            runtime.prepare_engine_args(args)
        return args

    def spawn(self, config):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        child = context.Process(target=_spawn_read, args=(queue, config))
        child.start()
        try:
            result = queue.get(timeout=30)
            child.join(timeout=15)
            self.assertFalse(child.is_alive())
            self.assertEqual(child.exitcode, 0)
            self.assertNotIn("error", result)
            return result
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
            queue.close()
            queue.join_thread()

    def test_head2_explicit_grouped_settings_override_environment_and_serialize(self):
        os.environ.update(MBX_DRAFT_HEAD2="missing.pt", MBX_DRAFT_HEAD2_K="invalid",
                          MBX_DRAFT_HEAD2_SCORER="invalid")
        with self.assertLogs("vllm.rtx_spark", level="INFO") as logs:
            args = self.prepare(optimizations={"draft_head2_basis": str(self.basis),
                "draft_head2_k": 512, "draft_head2_scorer": "fp4"})
        settings = args.additional_config["rtx_spark"]
        self.assertEqual(settings["draft_head2_k"], 512)
        self.assertEqual(settings["draft_head2_scorer"], "fp4")
        self.assertNotIn("optimizations", settings)
        qwen.validate(args, settings)
        self.assertEqual(os.environ["MBX_DRAFT_HEAD2"], str(self.basis.resolve()))
        self.assertEqual(os.environ["MBX_DRAFT_HEAD2_K"], "512")
        self.assertTrue(any("sources=" in line and "config" in line for line in logs.output))

    def test_environment_precedes_defaults_and_another_preparation_can_change_it(self):
        os.environ.update(MBX_DRAFT_HEAD2=str(self.basis), MBX_DRAFT_HEAD2_K="768",
                          MBX_DRAFT_HEAD2_SCORER="fp8")
        first = self.prepare()
        self.assertEqual(first.additional_config["rtx_spark"]["draft_head2_k"], 768)
        self.prepare(draft_head2_k=512)
        again = self.prepare()
        self.assertEqual(again.additional_config["rtx_spark"]["draft_head2_k"], 768)
        self.assertEqual(os.environ["MBX_DRAFT_HEAD2_K"], "768")

    def test_invalid_and_disabled_head2_settings_fail_before_initialization(self):
        cases = ({"draft_head2_k": 0}, {"draft_head2_k": True}, {"draft_head2_k": "512"},
                 {"draft_head2_basis": "missing.pt"}, {"draft_head2_scorer": "typo"},
                 {"draft_head2": False, "draft_head2_k": 512})
        for supplied in cases:
            args = NS(model="Qwen3.8-Flash-Next-FP4-Local", additional_config={"rtx_spark": supplied})
            with self.subTest(options=supplied), patch.object(bootstrap, "initialize") as initialize:
                with self.assertRaises((TypeError, ValueError)):
                    runtime.prepare_engine_args(args)
                initialize.assert_not_called()

    def test_real_spawn_reads_frozen_head2_and_preserved_external_policy(self):
        policy = self.directory / "depth.json"
        original = {"mode": "dynamic", "min": 2, "window": 96, "promote": [80, 70], "demote": [20, 10]}
        policy.write_text(json.dumps(original), encoding="utf-8")
        before = policy.read_bytes()
        os.environ["MBX_DEPTH_CFG"] = str(policy)
        args = self.prepare(draft_head2_basis=str(self.basis), draft_head2_k=1024,
                            draft_head2_scorer="fp8", draft_min_depth=3)
        result = self.spawn(args)
        self.assertEqual(result["head2"], {key: args.additional_config["rtx_spark"][key] for key in
            ("draft_head2_basis", "draft_head2_k", "draft_head2_scorer")})
        self.assertEqual(result["environment"], [str(self.basis.resolve()), "1024", "fp8"])
        self.assertEqual(result["policy"], dict(original, min=3))
        self.assertEqual(policy.read_bytes(), before)

    def test_real_spawn_opt_out_keeps_native_initialization_without_qwen_policy(self):
        args = NS(model="Qwen3.8-Flash-Next-FP4-Local",
                  additional_config={"rtx_spark": {"model": "none"}})
        runtime.prepare_engine_args(args)
        result = self.spawn(args)
        self.assertEqual(result, {"model": "none", "spawn": "spawn", "depth_disabled": True})
        self.assertNotIn("MBX_DEPTH_CFG", os.environ)

    def test_unreadable_or_nonobject_external_policy_is_not_silently_replaced(self):
        os.environ["MBX_DEPTH_CFG"] = str(self.directory / "missing.json")
        with self.assertRaises(FileNotFoundError):
            self.prepare(draft_min_depth=3)
        source = self.directory / "invalid.json"
        source.write_text("[]", encoding="utf-8")
        os.environ["MBX_DEPTH_CFG"] = str(source)
        with self.assertRaisesRegex(ValueError, "JSON object"):
            self.prepare(draft_min_depth=3)
        self.assertEqual(source.read_text(encoding="utf-8"), "[]")

    def test_nvfp4_storage_admits_other_geometries_and_preserves_known_baseline(self):
        self.assertEqual(core.expected_storage_bytes(248320, 2560), 357580804)
        self.assertEqual(core.expected_storage_bytes(128, 128), 9220)
        for shape in ((0, 128), (128, 0), (128, 127)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                core.expected_storage_bytes(*shape)


if __name__ == "__main__":
    unittest.main()
