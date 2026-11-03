"""Source-only regression for Mamba prefill checkpoint block units.

PR55601 already fixes the worker state seed; this tests the analogous remaining
Scheduler boundary bug. No Torch/model/native import or numerical inference.
Pass --site for a freshly composed source and optionally --baseline for before.
"""
import argparse
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument("--site",type=Path)
parser.add_argument("--baseline",type=Path)
ARGS,REST=parser.parse_known_args()
SITE=ARGS.site.resolve() if ARGS.site else None
BASE=ARGS.baseline.resolve() if ARGS.baseline else None
REL="vllm/v1/core/sched/scheduler.py"


def method(path,name,owner=None):
    tree=ast.parse(path.read_text(encoding="utf-8"))
    body=tree.body if owner is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    return next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)


def source_functions(site):
    helpers=[copy.deepcopy(method(site/"vllm/v1/kv_cache_interface.py",name)) for name in
        ("get_mamba_prefill_checkpoint_position","is_mamba_prefill_checkpoint_valid")]
    split=copy.deepcopy(method(site/REL,"_mamba_block_aligned_split","Scheduler"))
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*helpers,split],type_ignores=[])
    ast.fix_missing_locations(module)
    scope={"cdiv":lambda a,b:(a+b-1)//b}
    exec(compile(module,"source-extracted-mamba-checkpoint-metadata","exec"),scope)
    return scope


def state(mamba=6912,minimum=1728,unit=32):
    return NS(cache_config=NS(block_size=minimum,mamba_block_size=mamba),block_size=mamba,mamba_state_block_size=mamba,
        hash_block_size=unit,mamba_partial_cache_hit=True,mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=64 if unit in (32,64) else 192,
        use_eagle_block_drop=False,mamba_shared_prefix_checkpoint=False,
        max_num_scheduled_tokens=8192,scheduler_config=NS(long_prefill_token_threshold=0))


def request(prompt,start=0):
    return NS(num_computed_tokens=start,num_prompt_tokens=prompt,num_tokens=prompt,shared_prefix_boundary=0)


def chunks(split,scheduler,prompt):
    req=request(prompt);ends=[]
    while req.num_computed_tokens<prompt:
        count=split(scheduler,req,min(8192,prompt-req.num_computed_tokens))
        if count<=0:raise AssertionError("Production chunk splitter made no progress")
        req.num_computed_tokens+=count;ends.append(req.num_computed_tokens)
    return ends


@unittest.skipUnless(SITE,"Pass --site for actual source-extracted scheduler regression")
class Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.scope=source_functions(SITE);cls.split=staticmethod(cls.scope["_mamba_block_aligned_split"])

    def test_native_K7_mismatch_requires_the_8704_boundary(self):
        scheduler=state();req=request(8721,6912)
        # At the actual Mamba block size, the reserved checkpoint column would
        # equal the running source column, so the export is not valid here.
        valid=self.scope["is_mamba_prefill_checkpoint_valid"]
        self.assertFalse(valid(6912,8721,8704,32,6912,64))
        self.assertTrue(valid(6912,8721,8704,32,1728,64))
        self.assertEqual(self.split(scheduler,req,8721-6912),8704-6912)
        self.assertEqual(chunks(self.split,scheduler,8721),[6912,8704,8721])

    def test_before_source_reproduces_skipped_checkpoint(self):
        if BASE:
            old=source_functions(BASE)["_mamba_block_aligned_split"]
        else:
            # This isolated counterfactual only reintroduces the original
            # offending unit; the production splitter body stays unmodified.
            node=copy.deepcopy(method(SITE/REL,"_mamba_block_aligned_split","Scheduler"))
            node.body=[n for n in node.body if not (isinstance(n,ast.Assert) and ast.unparse(n.test)=="block_size is not None")]
            for n in ast.walk(node):
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="block_size" for t in n.targets):
                    n.value=ast.parse("self.cache_config.block_size",mode="eval").body
            mod=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[]))
            scope=dict(self.scope);exec(compile(mod,"old-unit-only-counterfactual","exec"),scope)
            old=scope["_mamba_block_aligned_split"]
        self.assertEqual(old(state(),request(8721,6912),1809),1809)
        self.assertNotIn(8704,chunks(old,state(),8721))

    def test_minimum_other_group_block_cannot_change_mamba_chunks(self):
        for mamba in (3456,6912):
            for unit in (32,64,96):
                for prompt in (1098,8721,17442):
                    reference=chunks(self.split,state(mamba,mamba,unit),prompt)
                    for minimum in (16,64,864,1728,3456):
                        with self.subTest(mamba=mamba,unit=unit,prompt=prompt,minimum=minimum):
                            self.assertEqual(chunks(self.split,state(mamba,minimum,unit),prompt),reference)
                    boundary=(prompt-1)//unit*unit
                    # A reusable checkpoint is either a real chunk endpoint,
                    # or can be exported from its actual source/destination
                    # Mamba columns under the production helper's predicate.
                    start=0;covered=False
                    for end in reference:
                        if boundary==end or self.scope["is_mamba_prefill_checkpoint_valid"](
                                start,end,boundary,unit,mamba,64 if unit in (32,64) else 192):covered=True
                        start=end
                    self.assertTrue(covered,(mamba,unit,prompt,reference,boundary))

    def test_existing_equal_block_MTP_behavior_is_unchanged(self):
        if BASE:
            old=source_functions(BASE)["_mamba_block_aligned_split"]
        else:
            self.skipTest("Pass --baseline for unchanged equal-block branch comparison")
        for block in (3456,6912):
            for unit in (32,64,96):
                for prompt in (1098,8721,17442):
                    value=state(block,block,unit)
                    self.assertEqual(chunks(self.split,value,prompt),chunks(old,value,prompt))

    def test_factory_input_and_resolved_scheduler_unit_are_explicit(self):
        factory=method(SITE/"vllm/model_executor/layers/mamba/abstract.py","get_kv_cache_spec","MambaBase")
        source=ast.unparse(factory)
        self.assertIn("mamba_block_size = vllm_config.cache_config.mamba_block_size",source)
        self.assertIn("block_size=mamba_block_size",source)
        splitter=ast.unparse(method(SITE/REL,"_mamba_block_aligned_split","Scheduler"))
        self.assertIn("block_size = self.mamba_state_block_size",splitter)
        self.assertNotIn("block_size = self.cache_config.mamba_block_size",splitter)
        constructor=ast.unparse(method(SITE/REL,"__init__","Scheduler"))
        self.assertIn("manager.block_size",constructor)
        self.assertIn("self.kv_cache_manager.coordinator.single_type_managers",constructor)
        self.assertIn("assert block_size is not None",splitter)
        self.assertNotIn("block_size = self.cache_config.block_size",splitter)

    def test_only_resolved_derivation_and_splitter_unit_change(self):
        if BASE is None:self.skipTest("Pass --baseline for whole-module AST comparison")
        old=ast.parse((BASE/REL).read_text(encoding="utf-8"))
        new=ast.parse((SITE/REL).read_text(encoding="utf-8"))
        cls=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=="Scheduler")
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
        init.body=[n for n in init.body if not (
            (isinstance(n,ast.Assign) and any(
                (isinstance(t,ast.Name) and t.id=="mamba_state_block_sizes") or
                (isinstance(t,ast.Attribute) and t.attr=="mamba_state_block_size") for t in n.targets))
            or (isinstance(n,ast.If) and "len(mamba_state_block_sizes)" in ast.unparse(n)))]
        split=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_mamba_block_aligned_split")
        split.body=[n for n in split.body if not (isinstance(n,ast.Assert) and ast.unparse(n.test)=="block_size is not None")]
        for n in ast.walk(split):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="block_size" for t in n.targets):
                n.value=ast.parse("self.cache_config.block_size",mode="eval").body
        # The historical pre-bug baseline also predates the independently
        # tested DFlash drop-policy correction. Reverse that exact warning
        # predicate, retaining full-module equality for every other statement.
        drop_test=ast.parse("speculative_config.needs_eagle_block_drop() and speculative_config.disable_eagle_block_drop",mode="eval").body
        matches=[n for n in ast.walk(init) if isinstance(n,ast.If)
            and ast.dump(n.test,include_attributes=False)==ast.dump(drop_test,include_attributes=False)]
        self.assertEqual(len(matches),1)
        matches[0].test=ast.parse("self.use_eagle and not self.use_eagle_block_drop",mode="eval").body
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))


if __name__=="__main__":unittest.main(argv=[sys.argv[0]]+REST,verbosity=2)
