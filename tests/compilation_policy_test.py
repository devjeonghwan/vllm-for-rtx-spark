"""Compilation policy is resolved before vLLM decisions and replayed in spawn."""

from contextlib import ExitStack
from dataclasses import replace
from functools import cache
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()

from _rtx_spark import plugin, recipes, runtime  # noqa: E402
from _rtx_spark.optimizations import resolve_options  # noqa: E402
from _rtx_spark.optimizations import compilation as policy  # noqa: E402
from _rtx_spark.runtime import bootstrap  # noqa: E402


def _register_fixture():
    recipes.register_model("policy-fixture", lambda args, options: None,
                           lambda args, options: None, options=(policy.BREAKABLE_CUDAGRAPH,))


def _spawn_policy(queue, config):
    try:
        _register_fixture()
        # Bootstrap may run repeatedly before the worker receives VllmConfig.
        os.environ[policy._ENVIRONMENT] = "1" if not config.additional_config["rtx_spark"][policy._OPTION] else "0"
        bootstrap.initialize()
        bootstrap.initialize()
        before = os.environ[policy._ENVIRONMENT]
        policy.prepare_compilation_policy(config)
        bootstrap.initialize()
        queue.put({"before": before, "effective": policy.effective_compilation_policy(config),
                   "torch_imported": "torch" in sys.modules, "vllm_imported": "vllm" in sys.modules})
    except BaseException as error:
        queue.put({"error": repr(error)})


class CompilationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ))
        os.environ.pop(policy._ENVIRONMENT, None)
        os.environ.pop("TORCH_COMPILE_DISABLE", None)
        self.stack.enter_context(patch.object(policy, "_projection", None))
        self.stack.enter_context(patch.dict(recipes._MODELS, {}, clear=True))
        self.stack.enter_context(patch.object(recipes, "register_builtin_models"))
        self.stack.enter_context(patch.object(plugin, "register"))
        _register_fixture()

    @staticmethod
    def args(options=None, *, mode=None, backend="inductor", model="policy-fixture"):
        return SimpleNamespace(model="checkpoint", additional_config={"rtx_spark": dict(options or {}, model=model)},
                               compilation_config=SimpleNamespace(mode=mode, backend=backend,
                                                                  cudagraph_mode="FULL_AND_PIECEWISE",
                                                                  custom_ops=["all"]))

    def test_recipe_default_preserves_qwen_behavior_without_global_bootstrap_default(self):
        args = self.args()
        runtime.prepare_engine_args(args)
        self.assertEqual(os.environ[policy._ENVIRONMENT], "1")
        self.assertTrue(args.additional_config["rtx_spark"][policy._OPTION])
        self.assertIsNone(args.compilation_config.mode)
        from _rtx_spark.recipes.qwen3_8_flash_next import OPTIONS

        self.assertTrue(resolve_options({}, OPTIONS)[policy._OPTION])
        self.assertNotIn('"VLLM_USE_BREAKABLE_CUDAGRAPH":',
                         (ROOT / "src/runtime/bootstrap.py").read_text(encoding="utf-8"))

    def test_caller_environment_overrides_recipe_and_explicit_config_overrides_environment(self):
        for env, supplied, expected in (("0", {}, False), ("1", {}, True),
                                        ("1", {policy._OPTION: False}, False),
                                        ("0", {"optimizations": {policy._OPTION: True}}, True)):
            with self.subTest(env=env, supplied=supplied), patch.object(policy, "_projection", None):
                os.environ[policy._ENVIRONMENT] = env
                args = self.args(supplied)
                runtime.prepare_engine_args(args)
                self.assertIs(args.additional_config["rtx_spark"][policy._OPTION], expected)
                self.assertEqual(os.environ[policy._ENVIRONMENT], "1" if expected else "0")

    def test_owned_environment_does_not_replace_the_next_recipe_default(self):
        first = self.args()
        runtime.prepare_engine_args(first)
        recipes.register_model("other-policy", lambda args, options: None, lambda args, options: None,
                               options=(replace(policy.BREAKABLE_CUDAGRAPH, default=False),))
        second = self.args(model="other-policy")
        runtime.prepare_engine_args(second)
        self.assertFalse(second.additional_config["rtx_spark"][policy._OPTION])
        self.assertEqual(os.environ[policy._ENVIRONMENT], "0")

    def test_invalid_environment_is_rejected_unless_explicit_config_supersedes_it(self):
        os.environ[policy._ENVIRONMENT] = "false"
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            runtime.prepare_engine_args(self.args())
        runtime.prepare_engine_args(self.args({policy._OPTION: False}))
        self.assertEqual(os.environ[policy._ENVIRONMENT], "0")

    def test_conflicting_native_compile_mode_fails_before_recipe_mutation(self):
        for mode in (1, 2, 3, "VLLM_COMPILE"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "conflicts.*mode"):
                runtime.prepare_engine_args(self.args(mode=mode))
            self.assertNotIn(policy._ENVIRONMENT, os.environ)
        runtime.prepare_engine_args(self.args({policy._OPTION: False}, mode=3))
        self.assertEqual(os.environ[policy._ENVIRONMENT], "0")

    def test_native_mode_backend_and_graph_scope_are_separate(self):
        args = self.args({policy._OPTION: False}, mode=0, backend="eager")
        runtime.prepare_engine_args(args)
        self.assertEqual(policy.effective_compilation_policy(args), {
            "breakable_cudagraph": False, "breakable_environment": "0", "mode": 0,
            "backend": "eager", "cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]})
        self.assertEqual(args.compilation_config.cudagraph_mode, "FULL_AND_PIECEWISE")

    def test_explicit_compile_rejects_other_compile_disable_switches(self):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        with self.assertRaisesRegex(ValueError, "disables it"):
            runtime.prepare_engine_args(self.args({policy._OPTION: False}, mode=3))
        os.environ.pop("TORCH_COMPILE_DISABLE")
        args = self.args({policy._OPTION: False}, mode=3)
        args.enforce_eager = True
        with self.assertRaisesRegex(ValueError, "disables it"):
            runtime.prepare_engine_args(args)

    def test_direct_config_compile_conflict_reads_model_config_enforce_eager(self):
        config = self.args({policy._OPTION: False}, mode=3)
        config.model_config = SimpleNamespace(enforce_eager=True)
        with self.assertRaisesRegex(ValueError, "enforce_eager.*disables it"):
            policy.prepare_compilation_policy(config)
        self.assertNotIn(policy._ENVIRONMENT, os.environ)
        self.assertEqual(config.compilation_config.mode, 3)
        # A direct config that explicitly disables compilation remains valid.
        config.compilation_config.mode = 0
        policy.prepare_compilation_policy(config)
        self.assertEqual(os.environ[policy._ENVIRONMENT], "0")

    def test_types_and_flat_grouped_conflicts_remain_central(self):
        for value in (0, "false", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                runtime.prepare_engine_args(self.args({policy._OPTION: value}))
        with self.assertRaisesRegex(ValueError, "Conflicting RTX Spark setting"):
            runtime.prepare_engine_args(self.args({policy._OPTION: False,
                                                  "optimizations": {policy._OPTION: True}}))

    def test_replay_clears_an_existing_vllm_env_cache_without_replacing_the_reader(self):
        envs = ModuleType("vllm.envs")
        envs.__getattr__ = cache(lambda name: os.environ.get(name))
        os.environ[policy._ENVIRONMENT] = "1"
        reader = envs.__getattr__
        self.assertEqual(reader(policy._ENVIRONMENT), "1")
        with patch.dict(sys.modules, {"vllm.envs": envs}):
            runtime.prepare_engine_args(self.args({policy._OPTION: False}, mode=3))
        self.assertIs(envs.__getattr__, reader)
        self.assertEqual(reader(policy._ENVIRONMENT), "0")

    def test_unprepared_no_recipe_config_does_not_apply_a_qwen_default(self):
        config = SimpleNamespace(model="ordinary-model", additional_config={})
        policy.prepare_compilation_policy(config)
        self.assertNotIn(policy._ENVIRONMENT, os.environ)
        self.assertEqual(config.additional_config, {})
        os.environ[policy._ENVIRONMENT] = "0"
        policy.prepare_compilation_policy(config)
        self.assertEqual(os.environ[policy._ENVIRONMENT], "0")

    def test_no_match_engine_restores_only_our_earlier_projection(self):
        runtime.prepare_engine_args(self.args())
        config = SimpleNamespace(model="ordinary-model", additional_config={})
        runtime.prepare_engine_args(config)
        self.assertNotIn(policy._ENVIRONMENT, os.environ)
        runtime.prepare_engine_args(self.args())
        policy.prepare_compilation_policy(config)
        self.assertNotIn(policy._ENVIRONMENT, os.environ)
        runtime.prepare_engine_args(self.args())
        os.environ[policy._ENVIRONMENT] = "0"  # A new caller setting is preserved.
        policy.prepare_compilation_policy(config)
        self.assertEqual(os.environ[policy._ENVIRONMENT], "0")

    def test_replay_preserves_prepared_no_match_when_hf_architecture_appears_later(self):
        recipes.register_model("late-policy", lambda args, options: None, lambda args, options: None,
                               architectures=("LateArchitecture",), options=(policy.BREAKABLE_CUDAGRAPH,))
        args = SimpleNamespace(model="renamed-checkpoint", additional_config={})
        runtime.prepare_engine_args(args)
        config = SimpleNamespace(model_config=SimpleNamespace(model="renamed-checkpoint",
                                                               hf_config=SimpleNamespace(
                                                                   architectures=["LateArchitecture"])),
                                 additional_config={})
        policy.prepare_compilation_policy(config)
        runtime.configure_engine(config, engine_args=args)
        self.assertEqual(config.additional_config, {})
        self.assertNotIn(policy._ENVIRONMENT, os.environ)

    def test_direct_config_uses_an_explicit_recipe_without_automatic_reselection(self):
        config = self.args({policy._OPTION: False}, mode=3)
        policy.prepare_compilation_policy(config)
        self.assertEqual(os.environ[policy._ENVIRONMENT], "0")
        self.assertFalse(config.additional_config["rtx_spark"][policy._OPTION])

    def test_prepared_policy_replays_in_a_real_spawn_before_model_imports(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = self.args({policy._OPTION: enabled}, mode=0 if enabled else 3)
                runtime.prepare_engine_args(config)
                context = multiprocessing.get_context("spawn")
                queue = context.Queue()
                child = context.Process(target=_spawn_policy, args=(queue, config))
                child.start()
                try:
                    result = queue.get(timeout=30)
                    child.join(timeout=15)
                    self.assertFalse(child.is_alive())
                    self.assertEqual(child.exitcode, 0)
                    self.assertNotIn("error", result)
                    self.assertEqual(result["before"], "0" if enabled else "1")
                    self.assertEqual(result["effective"]["breakable_environment"], "1" if enabled else "0")
                    self.assertIs(result["effective"]["breakable_cudagraph"], enabled)
                    self.assertEqual(result["effective"]["mode"], 0 if enabled else 3)
                    self.assertFalse(result["torch_imported"])
                    self.assertFalse(result["vllm_imported"])
                finally:
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=10)
                    queue.close()
                    queue.join_thread()

    def test_source_hooks_precede_upstream_auto_selection_and_worker_class_import(self):
        source = (ROOT / "patches/vllm/0012-recipe-compilation-policy.patch").read_text(encoding="utf-8")
        self.assertIn("prepare_compilation_policy(self)", source)
        self.assertIn("prepare_compilation_policy(vllm_config)", source)
        self.assertLess(source.index("prepare_compilation_policy(self)"),
                        source.index("self.instance_id ="))
        self.assertLess(source.index("prepare_compilation_policy(vllm_config)"),
                        source.index("from vllm.plugins import load_general_plugins"))
        manifest = json.loads((ROOT / "patches/vllm/manifest.json").read_text(encoding="utf-8"))
        self.assertIn("0012-recipe-compilation-policy.patch", manifest["patches"])


if __name__ == "__main__":
    unittest.main()
