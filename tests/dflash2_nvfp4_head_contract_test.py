"""CPU contracts for the opt-in, draft-only original NVFP4 head boundary.

These checks cover packaging, options, late-load order and alias ownership.
They do not execute quantization/CUDA or claim BF16/NVFP4 numerical identity.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import pickle
import runpy
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--site", type=Path)
ARGS, REST = parser.parse_known_args()
runpy.run_path(ROOT / "scripts/load_source_package.py")["load"]()
from _rtx_spark.recipes import qwen3_8 as recipe
from _rtx_spark.recipes.qwen3_8 import dflash_nvfp4_core as core
from _rtx_spark.recipes.qwen3_8 import dflash_nvfp4_head as head
from _rtx_spark.runtime.vllm.adapter import bind_recipe

REL = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
UTILS = "vllm.v1.worker.gpu.spec_decode.eagle.utils"
OWNER = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
CORE_AST_SHA = "fd1a0f5d34816123ef7a7ce453083ef9dac4c1d4483b990881a0763dc3c19755"
HEAD_AST_SHA = "0d4cc61403625cb3d034141d630a9a6706179c96bee80c8c68e1b33215fb4df7"
PRESERVED = ("tensor_metadata", "module_metadata", "target_tree_metadata", "tree_summary", "hash_weight", "convert")


def ast_fingerprint(path, selected=None):
    nodes = {node.name: ast.dump(node, include_attributes=False)
             for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    if selected is not None:
        nodes = {name: nodes[name] for name in selected}
    return hashlib.sha256(json.dumps(nodes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config(enabled=True):
    return NS(additional_config={"rtx_spark": {"dflash2_nvfp4_head": enabled}},
        speculative_config=NS(method="dflash", draft_model_config=NS(hf_config=NS(architectures=["DFlash2DraftModel"]))),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           decode_context_parallel_size=1, prefill_context_parallel_size=1),
        model_config=NS(hf_config=NS(architectures=["Qwen4ExpForConditionalGeneration"])))


class HeadContractTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("_LOAD_OWNER", None), ("_LATE_HOOK_CALLED", False),
                            ("_OWNERS", None), ("_ACTIVATION", None), ("_FAILURE", None)):
            p = patch.object(head, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_bool_default_and_grouped_options_are_strict(self):
        self.assertIs(recipe._validate_options({})["dflash2_nvfp4_head"], False)
        self.assertIs(recipe._validate_options({"optimizations": {"dflash2_nvfp4_head": True}})["dflash2_nvfp4_head"], True)
        for value in (0, 1, "true", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                recipe._validate_options({"dflash2_nvfp4_head": value})

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

    def test_validated_core_math_and_target_guards_preserved(self):
        self.assertEqual(ast_fingerprint(Path(core.__file__)), CORE_AST_SHA)
        self.assertEqual(ast_fingerprint(Path(head.__file__), PRESERVED), HEAD_AST_SHA)
        self.assertIn("from . import dflash_nvfp4_core as core", Path(head.__file__).read_text(encoding="utf-8"))
        self.assertNotIn("research/results", Path(head.__file__).read_text(encoding="utf-8"))

    def test_prepared_loader_preserves_original_load_and_exact_late_return(self):
        if ARGS.site:
            tree = ast.parse((ARGS.site / REL).read_text(encoding="utf-8"))
            owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DFlashSpeculator")
            method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "load_draft_model")
            body = ast.Module(body=method.body, type_ignores=[])
        else:
            patch_source = (ROOT / "patches/vllm/0010-dflash2-nvfp4-head.patch").read_text(encoding="utf-8")
            added = "\n".join(line[1:] for line in patch_source.splitlines() if line.startswith("+") and not line.startswith("+++"))
            import textwrap
            body = ast.parse(textwrap.dedent(added))
        expected = ast.parse('draft_model = load_dflash_model(target_model, self.vllm_config)\nfrom vllm._rtx_spark.recipes.qwen3_8.dflash_nvfp4_head import maybe_convert\nreturn maybe_convert(self, draft_model, target_model)')
        self.assertEqual(ast.dump(body, include_attributes=False), ast.dump(expected, include_attributes=False))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *REST])
