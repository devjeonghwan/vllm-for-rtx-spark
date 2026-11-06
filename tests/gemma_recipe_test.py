"""The second recipe validates Gemma scope without entering Qwen preparation."""

from contextlib import ExitStack
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as Namespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()

from _rtx_spark import recipes, runtime  # noqa: E402
from _rtx_spark.optimizations import resolve_options  # noqa: E402
from _rtx_spark.optimizations import compilation  # noqa: E402
from _rtx_spark.recipes import gemma4_12b_fp8 as gemma  # noqa: E402
from _rtx_spark.runtime import bootstrap  # noqa: E402


def _metadata():
    return {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "text_config": {"hidden_size": 3840, "num_hidden_layers": 48, "enable_moe_block": False},
        "quantization_config": {
            "quant_method": "compressed-tensors", "format": "float-quantized", "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {"type": "float", "num_bits": 8, "strategy": "channel", "dynamic": False,
                            "symmetric": True, "group_size": None},
                "input_activations": {"type": "float", "num_bits": 8, "strategy": "token", "dynamic": True,
                                      "symmetric": True, "group_size": None},
            }},
        },
    }


def _args(options=None):
    return Namespace(model=gemma.MODEL_ID, language_model_only=True,
                     load_format="safetensors", speculative_config=None,
                     additional_config={"rtx_spark": dict(options or {}, model="gemma4-12b-fp8")},
                     compilation_config=Namespace(mode=0, backend="inductor", cudagraph_mode="FULL_AND_PIECEWISE"))


def _config(options=None):
    args = _args(options)
    return Namespace(additional_config=args.additional_config, speculative_config=None,
                     model_config=Namespace(model=gemma.MODEL_ID, hf_config=Namespace(**_metadata()),
                                            multimodal_config=Namespace(language_model_only=True, enable_mm_embeds=False)),
                     parallel_config=Namespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                                               decode_context_parallel_size=1, prefill_context_parallel_size=1),
                     load_config=Namespace(load_format="safetensors"), compilation_config=args.compilation_config)


def _spawn_worker(queue, config):
    try:
        bootstrap.initialize()
        compilation.prepare_compilation_policy(config)
        runtime.configure_engine(config)
        queue.put({"options": config.additional_config["rtx_spark"],
                   "breakable": os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH"),
                   "qwen_env": sorted(name for name in os.environ
                                      if name.startswith(("MBX_DRAFT", "MBX_DEPTH", "MBX_PLE"))),
                   "qwen_helpers": sorted(name for name in sys.modules if ".qwen3_8_flash_next.helpers" in name),
                   "torch": "torch" in sys.modules, "vllm": "vllm" in sys.modules})
    except BaseException as error:
        queue.put({"error": repr(error)})


class GemmaRecipeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ))
        self.stack.enter_context(patch.dict(recipes._MODELS, dict(recipes._MODELS), clear=True))
        self.stack.enter_context(patch.object(compilation, "_projection", None))
        for name in list(os.environ):
            if name.startswith(("MBX_DRAFT", "MBX_DEPTH", "MBX_PLE")) or name == "VLLM_USE_BREAKABLE_CUDAGRAPH":
                os.environ.pop(name)

    def test_builtin_registry_uses_upstream_architecture_without_replacement(self):
        recipes.register_builtin_models()
        self.assertEqual(recipes.resolve_model_name(_args()), "gemma4-12b-fp8")
        self.assertTrue(gemma.matches(Namespace(model=gemma.MODEL_ID.lower())))
        self.assertFalse(gemma.matches(Namespace(model="Qwen3.8-Flash-Next-FP4-Local")))
        self.assertEqual(gemma.ARCHITECTURES, ("Gemma4UnifiedForConditionalGeneration",))
        self.assertEqual(gemma.MODEL_CLASSES, {})

    def test_initial_options_keep_compile_and_fused_optimizations_off(self):
        resolved = resolve_options({"model": "gemma4-12b-fp8"}, gemma.OPTIONS)
        self.assertTrue(resolved["breakable_cudagraph"])
        self.assertFalse(resolved["fused_fp8"])
        for invalid in ("false", 0, None):
            with self.subTest(value=invalid), self.assertRaises(TypeError):
                resolve_options({"fused_fp8": invalid}, gemma.OPTIONS)
        with self.assertRaisesRegex(ValueError, "Unknown RTX Spark setting"):
            resolve_options({"draft_fp8": True}, gemma.OPTIONS)

    def test_prepare_does_not_call_qwen_or_mutate_checkpoint_fields(self):
        from _rtx_spark.recipes import qwen3_8_flash_next

        args = _args()
        fields = deepcopy(vars(args))
        with patch.object(bootstrap, "initialize"), patch.object(qwen3_8_flash_next, "prepare", side_effect=AssertionError("Qwen")):
            runtime.prepare_engine_args(args)
        self.assertEqual(args.model, fields["model"])
        self.assertEqual(args.load_format, "safetensors")
        self.assertFalse(hasattr(args, "hf_overrides"))
        self.assertEqual(os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"], "1")
        self.assertFalse(any(name.startswith(("MBX_DRAFT", "MBX_DEPTH", "MBX_PLE")) for name in os.environ))

    def test_final_config_checks_actual_quantization_and_architecture(self):
        gemma.validate(_config(), {})
        for mutate, message in (
            (lambda hf: hf.update(model_type="gemma4"), "Unified"),
            (lambda hf: hf["text_config"].update(hidden_size=4096), "12B dense"),
            (lambda hf: hf["text_config"].update(enable_moe_block=True), "12B dense"),
            (lambda hf: hf["quantization_config"].update(quant_method="modelopt"), "compressed-tensors"),
            (lambda hf: hf["quantization_config"]["config_groups"]["group_0"]["weights"].update(strategy="block"), "per-channel"),
            (lambda hf: hf["quantization_config"]["config_groups"]["group_0"]["input_activations"].update(dynamic=False), "per-token"),
        ):
            config = _config()
            metadata = _metadata()
            mutate(metadata)
            config.model_config.hf_config = Namespace(**metadata)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                gemma.validate(config, {})

    def test_supported_scope_is_checked_before_generic_startup(self):
        for name, value in (("language_model_only", False), ("speculative_config", {"method": "mtp"}),
                            ("use_replayssm", True), ("enable_mm_embeds", True), ("tensor_parallel_size", 2)):
            args = _args()
            setattr(args, name, value)
            with self.subTest(field=name), patch.object(bootstrap, "initialize") as initialize:
                with self.assertRaises(NotImplementedError):
                    gemma.prepare(args, {})
                initialize.assert_not_called()
        args = _args()
        args.load_format = "mbx"
        with self.assertRaisesRegex(ValueError, "Qwen-specific"):
            gemma.prepare(args, {})

    def test_final_scope_uses_nested_native_config_fields(self):
        for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                      "decode_context_parallel_size", "prefill_context_parallel_size"):
            config = _config()
            setattr(config.parallel_config, field, 2)
            with self.subTest(field=field), self.assertRaisesRegex(NotImplementedError, field):
                gemma.validate(config, {})
        config = _config()
        config.model_config.multimodal_config.language_model_only = False
        with self.assertRaisesRegex(NotImplementedError, "language-model-only"):
            gemma.validate(config, {})

    def test_initial_profile_keeps_its_own_kv_budget_and_official_template(self):
        text = (ROOT / "configs/gemma4-12b-fp8/text.yaml").read_text(encoding="utf-8")
        profile = yaml.safe_load(text)
        self.assertEqual(profile["revision"], gemma.MODEL_REVISION)
        self.assertEqual(profile["tokenizer-revision"], gemma.MODEL_REVISION)
        self.assertEqual(profile["load-format"], "safetensors")
        self.assertEqual(profile["kv-cache-memory-bytes"], 8_000_000_000)
        self.assertEqual(profile["kv-cache-dtype"], "bfloat16")
        self.assertEqual(profile["max-model-len"], 32768)
        self.assertEqual(profile["max-num-seqs"], 16)
        self.assertTrue(profile["language-model-only"])
        self.assertFalse(profile["enable-prefix-caching"])
        self.assertEqual(profile["additional-config"], {"rtx_spark": {"model": "gemma4-12b-fp8",
                         "breakable_cudagraph": True, "fused_fp8": False}})
        self.assertEqual(profile["compilation-config"]["mode"], 0)
        self.assertTrue({"speculative-config", "hf-overrides", "tool-call-parser", "reasoning-parser",
                         "chat-template", "use-replayssm", "cpu-offload-gb"}.isdisjoint(profile))
        self.assertNotIn("MBX_", "\n".join(line for line in text.splitlines() if not line.startswith("#")))

    def test_selected_recipe_policy_replays_in_a_real_spawn_without_qwen_environment(self):
        args = _args()
        with patch.object(bootstrap, "initialize"):
            runtime.prepare_engine_args(args)
        config = _config(args.additional_config["rtx_spark"])
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        child = context.Process(target=_spawn_worker, args=(queue, config))
        child.start()
        try:
            result = queue.get(timeout=30)
            child.join(timeout=15)
            self.assertEqual(child.exitcode, 0)
            self.assertFalse(child.is_alive())
            self.assertNotIn("error", result)
            self.assertEqual(result["options"]["model"], "gemma4-12b-fp8")
            self.assertEqual(result["breakable"], "1")
            self.assertEqual(result["qwen_env"], [])
            self.assertEqual(result["qwen_helpers"], [])
            self.assertFalse(result["torch"])
            self.assertFalse(result["vllm"])
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
            queue.close()
            queue.join_thread()


if __name__ == "__main__":
    unittest.main()
