"""CPU contracts for the opt-in, draft-only original NVFP4 head boundary.

These checks cover packaging, options, late-load order and alias ownership.
They do not execute quantization/CUDA or claim BF16/NVFP4 numerical identity.
"""
from __future__ import annotations
import ast
import copy
from pathlib import Path
import pickle
import runpy
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
from support import SOURCE as SITE
runpy.run_path(ROOT / "scripts/load_source_package.py")["load"]()
from _rtx_spark.recipes import qwen3_8_flash_next as recipe
from _rtx_spark.recipes.qwen3_8_flash_next import dflash_nvfp4_core as core
from _rtx_spark.recipes.qwen3_8_flash_next import dflash_nvfp4_head as head
from _rtx_spark.runtime.vllm.adapter import bind_recipe

REL = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
UTILS = "vllm.v1.worker.gpu.spec_decode.eagle.utils"
OWNER = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"




def config(enabled=True):
    return NS(additional_config={"rtx_spark": {"dflash2_nvfp4_head": enabled}},
        speculative_config=NS(method="dflash", draft_model_config=NS(hf_config=NS(architectures=["DFlash2DraftModel"]))),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           decode_context_parallel_size=1, prefill_context_parallel_size=1),
        model_config=NS(hf_config=NS(architectures=["Qwen4ExpForConditionalGeneration"])))


def quantizer_fixture(value):
    amax = value
    g = float(amax.max()) / (448.0 * 6.0)
    return value, g


class HeadContractTests(unittest.TestCase):
    def test_chunk_quantizer_overrides_only_global_scale_and_preserves_original(self):
        value = NS(max=lambda: 5376.0)
        original_code = quantizer_fixture.__code__
        chunk_quantizer = core.quantizer_with_global_scale(quantizer_fixture)
        self.assertEqual(quantizer_fixture(value), (value, 2.0))
        self.assertEqual(chunk_quantizer(value, 3.5), (value, 3.5))
        self.assertIs(quantizer_fixture.__code__, original_code)

    @unittest.skipUnless(SITE, 'Set VLLM_RTX_SPARK_SOURCE_DIR for actual loader integration')
    def test_actual_loader_loads_original_before_the_optional_head_conversion(self):
        tree = ast.parse((SITE / REL).read_text(encoding='utf-8'))
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DFlashSpeculator')
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == 'load_draft_model')
        calls = []
        target, draft, converted, configuration = object(), object(), object(), object()
        def load(observed_target, observed_config):
            calls.append(('load', observed_target, observed_config))
            return draft
        helper_name = 'vllm._rtx_spark.recipes.qwen3_8_flash_next.dflash_nvfp4_head'
        helper = ModuleType(helper_name)
        helper.maybe_convert = lambda observed_owner, observed_draft, observed_target: (
            calls.append(('convert', observed_owner, observed_draft, observed_target)) or converted)
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        scope = {'load_dflash_model': load}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])),
                     str(SITE / REL), 'exec'), scope)
        subject = NS(vllm_config=configuration)
        with patch.dict(sys.modules, {helper_name: helper}):
            self.assertIs(scope['load_draft_model'](subject, target, []), converted)
        self.assertEqual(calls, [('load', target, configuration), ('convert', subject, draft, target)])

    def test_live_target_mutation_fails_without_releasing_conversion_owners(self):
        target, holder, target_model = object(), object(), object()
        draft = NS(lm_head=holder)
        target_metadata, holder_metadata, tree_metadata = {'pointer': 1}, {'pointer': 2}, {'modules': 3}
        owners = {'target': target, 'draft': draft, 'holder': holder, 'whole_target': target_model,
                  'whole_target_tree_before': copy.deepcopy(tree_metadata)}
        activation = {'target_module_before': copy.deepcopy(target_metadata),
                      'holder_metadata': copy.deepcopy(holder_metadata)}
        def metadata(module):
            return copy.deepcopy(target_metadata if module is target else holder_metadata)
        with patch.object(head, '_OWNERS', owners), patch.object(head, '_ACTIVATION', activation), \
                patch.object(head, 'module_metadata', side_effect=metadata), \
                patch.object(head, 'target_tree_metadata', return_value=tree_metadata):
            self.assertTrue(head.status()['whole_target_tree_metadata_unchanged'])
            target_metadata['pointer'] = 7
            with self.assertRaisesRegex(RuntimeError, 'metadata or binding changed'):
                head.status()
            self.assertIs(head._OWNERS, owners)
            self.assertIs(draft.lm_head, holder)

    def setUp(self):
        for name, value in (("_LOAD_OWNER", None), ("_LATE_HOOK_CALLED", False),
                            ("_OWNERS", None), ("_ACTIVATION", None), ("_FAILURE", None)):
            p = patch.object(head, name, value)
            p.start()
            self.addCleanup(p.stop)


    def test_omitted_and_false_leave_original_draft_untouched(self):
        draft = object()
        for additional in ({}, {"rtx_spark": {"dflash2_nvfp4_head": False}},
                {"rtx_spark": {"model": "another-recipe", "other_recipe_key": True}},
                {"rtx_spark": {"model": "another-recipe", "other_recipe_key": True, "dflash2_nvfp4_head": False}},
                {"rtx_spark": {"optimizations": {"dflash2_nvfp4_head": False, "other_recipe_key": True}}}):
            owner = NS(vllm_config=NS(additional_config=additional))
            with patch.object(recipe, "_validate_options", side_effect=AssertionError("Qwen validation must not run")), patch.object(head, "convert", side_effect=AssertionError("conversion must not run")):
                self.assertIs(head.maybe_convert(owner, draft, object()), draft)
        self.assertIs(head.maybe_convert(NS(vllm_config=NS()), draft, object()), draft)
        self.assertFalse(head.status()["installed"])
        self.assertFalse(head.status()["strong_owners"])

    def test_loader_option_read_is_strict_and_handles_grouped_conflicts(self):
        draft = object()
        for value in (0, 1, "true", None):
            for mapping in ({"dflash2_nvfp4_head": value}, {"optimizations": {"dflash2_nvfp4_head": value}}):
                with self.subTest(value=value, mapping=mapping), self.assertRaises(TypeError):
                    head.maybe_convert(NS(vllm_config=NS(additional_config={"rtx_spark": mapping})), draft, object())
        for flat, nested in ((True, False), (False, True)):
            mapping = {"dflash2_nvfp4_head": flat, "optimizations": {"dflash2_nvfp4_head": nested}}
            with self.assertRaises(ValueError):
                head.maybe_convert(NS(vllm_config=NS(additional_config={"rtx_spark": mapping})), draft, object())

    def test_serialized_config_carries_option_without_parent_hook_or_env(self):
        original = config(True)
        original_options = copy.deepcopy(original.additional_config)
        bind_recipe(original, "qwen3.8-flash-next")
        worker = pickle.loads(pickle.dumps(original))
        self.assertTrue(recipe._validate_options(worker.additional_config["rtx_spark"])["dflash2_nvfp4_head"])
        self.assertEqual(original_options, {"rtx_spark": {"dflash2_nvfp4_head": True}})
        self.assertNotIn("install_startup", vars(head))

    def test_enabled_wrong_model_method_or_parallel_rejected_before_conversion(self):
        cases = []
        for field in ("tensor_parallel_size", "pipeline_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size"):
            value = config()
            setattr(value.parallel_config, field, 2)
            cases.append(value)
        wrong_method = config()
        wrong_method.speculative_config.method = "eagle"
        cases.append(wrong_method)
        wrong_draft = config()
        wrong_draft.speculative_config.draft_model_config.hf_config.architectures = ["DFlashDraftModel"]
        cases.append(wrong_draft)
        wrong_target = config()
        wrong_target.model_config.hf_config.architectures = ["OtherTarget"]
        cases.append(wrong_target)
        for value in cases:
            owner = type("DFlash2Speculator", (), {})()
            owner.vllm_config = value
            with self.subTest(config=value), patch.object(head, "convert", side_effect=AssertionError), self.assertRaises(ValueError):
                head.maybe_convert(owner, object(), object())

    def test_late_hook_converts_once_after_original_load_and_owns_no_wrapper(self):
        target, target_head = object(), object()
        draft = NS(lm_head=target_head)
        owner_class = type("DFlash2Speculator", (), {})
        owner = owner_class()
        owner.vllm_config = config()
        utils = ModuleType(UTILS)
        utils.get_target_lm_head = lambda first, language: target_head
        module = ModuleType(OWNER)
        class OriginalOwner:
            def load_draft_model(self, target_model, names):
                pass
        module.DFlashSpeculator = OriginalOwner
        called = []
        with patch.dict(sys.modules, {UTILS: utils, OWNER: module}), patch.object(head, "convert", side_effect=lambda *args: called.append(args)):
            self.assertIs(head.maybe_convert(owner, draft, target), draft)
            self.assertEqual(called, [(draft, target_head, target)])
            status = head.status()
            self.assertTrue(status["installed"])
            self.assertTrue(status["startup_hook_binding_intact"])
            self.assertIs(OriginalOwner.load_draft_model, head._LOAD_OWNER)
            OriginalOwner.load_draft_model = lambda *_: None
            with self.assertRaises(RuntimeError):
                head.status()

    def test_original_shared_aliases_replaced_without_mutating_target(self):
        target, replacement = object(), object()
        shared = NS(head=target)
        unrelated = NS(head=object())
        draft = NS(lm_head=target, model=NS(layers=[NS(shared_head=shared), NS(shared_head=unrelated)]))
        fake_torch = ModuleType("torch")
        fake_torch.nn = NS(ModuleDict=dict)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result, aliases = core.replace_draft_head(draft, target, factory=lambda observed: replacement if observed is target else None)
        self.assertIs(result, replacement)
        self.assertIs(draft.lm_head, replacement)
        self.assertIs(shared.head, replacement)
        self.assertIsNot(unrelated.head, replacement)
        self.assertEqual(aliases, 1)




if __name__ == "__main__":
    unittest.main()
