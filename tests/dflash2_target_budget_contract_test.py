"""Actual source-extracted graph candidate/dispatch and runner-prefix fixtures.

Only enum/config/descriptor metadata and tensor slicing are symbolically
represented; no Torch/CUDA is imported. This is not native recurrent rollback
or target-distribution validation.
"""
from __future__ import annotations
import argparse
import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
import hashlib
from itertools import groupby, product
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--site',type=Path)
parser.add_argument('--baseline',type=Path)
ARGS, REST = parser.parse_known_args()
SITE = ARGS.site.resolve() if ARGS.site else None
BASE = ARGS.baseline.resolve() if ARGS.baseline else None
REL = Path('vllm/v1/worker/gpu/cudagraph_utils.py')
QUERY_REL = Path('vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py')
CAPTURE = sorted(set([1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,21,24,28,32,36,40,48,56,64,72,80,88,96,104,112,120,128]) | {5*b for b in range(8,17)})



class Mode(IntEnum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2
    BOTH = 3
    DECODE_ONLY = 4
    def decode_mode(self):
        return Mode.FULL
    def mixed_mode(self):
        return Mode.PIECEWISE if self == Mode.BOTH else Mode.NONE
    def separate_routine(self):
        return True


@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int
    num_reqs: int | None = None
    uniform_token_count: int | None = None
    max_query_len: int | None = None
    num_active_loras: int = 0
    num_ubatches: int = 1


def functions(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    init = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'CudaGraphManager'
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == '_init_candidates')
    dispatch = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'CudaGraphManager'
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'dispatch')
    compatible = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_is_compatible')
    # Execute the exact upstream schedule resolver with only annotations deferred.
    schedule_path = SITE / 'vllm/v1/spec_decode/dynamic/utils.py'
    schedule = ast.parse(schedule_path.read_text(encoding='utf-8'))
    defs = [n for n in schedule.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    env = {'CUDAGraphMode': Mode, 'BatchExecutionDescriptor': Descriptor, '_mbx_depth': None,
           'defaultdict': defaultdict, 'product': product, 'groupby': groupby,
           'round_up': lambda n, size: ((n + size - 1) // size) * size}
    module = ast.fix_missing_locations(ast.Module(body=[future, *defs, compatible, init, dispatch], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), env)
    return env


def manager(path, fixed, budget=3, *, dynamic=True, method='dflash'):
    env = functions(path)
    config = SimpleNamespace(num_speculative_tokens_per_batch_size=[(1,7,7),(8,16,budget)],
                             uses_dynamic_speculative_decoding=lambda: dynamic, method=method)
    vllm = SimpleNamespace(speculative_config=config, num_speculative_tokens=7,
                           cache_config=SimpleNamespace(use_recoverssm=True))
    value = SimpleNamespace(vllm_config=vllm, decode_query_len=8, max_num_reqs=16,
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=CAPTURE,max_cudagraph_capture_size=128),
        cudagraph_mode=Mode.DECODE_ONLY if fixed else Mode.BOTH, varlen_decode=False,
        lora_capture_cases=[0], _capture_descs={}, _candidates={}, _graphs_captured=True,
        _maybe_ubatch_twin=lambda desc: None, _resolve_effective_loras=lambda count: count)
    if fixed:
        query_source = path.parents[4]/QUERY_REL
        query_tree = ast.parse(query_source.read_text(encoding='utf-8'))
        query_class = next(n for n in query_tree.body if isinstance(n,ast.ClassDef) and n.name=='DFlashCudaGraphManager')
        value._fixed_draft_query_width = any(isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
            and n.value.value is True and any(isinstance(t,ast.Name) and t.id=='_fixed_draft_query_width'for t in n.targets)
            for n in query_class.body)
    env['_init_candidates'](value)
    value.dispatch = lambda b, n, q: env['dispatch'](value,b,n,q,0)
    return value


class Tensor:
    def __init__(self, values):
        self.values = [list(row) for row in values]
    def __setitem__(self, key, value):
        rows, columns = key
        for i in range(len(self.values))[rows]:
            for j in range(len(self.values[i]))[columns]:
                self.values[i][j] = value


@unittest.skipUnless(SITE is not None, "Pass --site for source-extracted target/query Graph contracts")
class Contracts(unittest.TestCase):
    def test_old_query_incorrectly_inherits_target_schedule(self):
        if BASE is None:self.skipTest("Pass --baseline for the pre-fix regression")
        value = manager(BASE / REL, True)
        self.assertEqual({d.uniform_token_count for d in value._capture_descs[Mode.FULL]}, {4,8})

    def test_target_schedule4and8_query_only8_all_B1to16(self):
        for budget in (3,4):
            target = manager(SITE / REL, False, budget)
            query = manager(SITE / REL, True, budget)
            self.assertEqual({d.uniform_token_count for d in target._capture_descs[Mode.FULL]}, {budget+1,8})
            self.assertEqual({d.uniform_token_count for d in query._capture_descs[Mode.FULL]}, {8})
            for b in range(1,17):
                width = 8 if b < 8 else budget+1
                a = target.dispatch(b,b*width,width)
                z = query.dispatch(b,b*8,8)
                self.assertEqual((a.cg_mode,a.uniform_token_count),(Mode.FULL,width))
                self.assertGreaterEqual(a.num_reqs,b)
                self.assertEqual(a.num_tokens,a.num_reqs*width)
                compatible=[d for d in target._capture_descs[Mode.FULL]
                            if d.uniform_token_count==width and d.num_reqs>=b and d.num_tokens>=b*width]
                self.assertEqual(a.num_tokens,min(d.num_tokens for d in compatible))
                if b==16:
                    self.assertEqual((a.num_reqs,a.num_tokens),(16,16*width))
                self.assertEqual((z.cg_mode,z.num_reqs,z.num_tokens,z.uniform_token_count),(Mode.FULL,b,b*8,8))

    def test_K4_target_capture_sizes_close_the_55row_gap(self):
        target=manager(SITE/REL,False,4)
        query=manager(SITE/REL,True,4)
        for b in range(1,17):
            width=8 if b<8 else 5
            selected=target.dispatch(b,b*width,width)
            draft=query.dispatch(b,b*8,8)
            self.assertEqual((selected.cg_mode,selected.num_reqs,selected.num_tokens),(Mode.FULL,b,b*width))
            self.assertEqual((draft.cg_mode,draft.num_reqs,draft.num_tokens),(Mode.FULL,b,b*8))

    def test_target_and_MTP_initializer_unchanged(self):
        if BASE is None:self.skipTest("Pass --baseline for target and MTP branch equality")
        for dynamic in (False,True):
            for method in ('mtp','dflash'):
                a = manager(BASE/REL,False,dynamic=dynamic,method=method)
                b = manager(SITE/REL,False,dynamic=dynamic,method=method)
                self.assertEqual(a._capture_descs,b._capture_descs)
                self.assertEqual(a._candidates,b._candidates)

    def test_original_actual_runner_tail_mask_keeps_all_q_prefixes(self):
        path = SITE/'vllm/v1/worker/gpu/model_runner.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='GPUModelRunner'
                        for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='sample_tokens')
        node = next(n for n in ast.walk(function) if isinstance(n,ast.If)
                    and ast.unparse(n.test)=='num_spec_tokens < self.num_speculative_steps')
        module = ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
        slots = [4,1,3]
        q = {(slot,step): (slot*100+step) for slot in slots for step in range(7)}
        for budget in (0,1,3,4,7):
            draft = Tensor([[slot*100+step for step in range(7)] for slot in slots])
            env={'draft_tokens':draft,'num_spec_tokens':budget,'self':SimpleNamespace(num_speculative_steps=7)}
            exec(compile(module,str(path),'exec'),env)
            self.assertEqual([row[:budget] for row in draft.values],[[q[slot,step]for step in range(budget)]for slot in slots])
            self.assertTrue(all(value==-1 for row in draft.values for value in row[budget:]))
        # No q storage or per-request7-row stride is changed by the patch.
        self.assertEqual(q,{(slot,step):(slot*100+step)for slot in slots for step in range(7)})


if __name__=='__main__':
    unittest.main(argv=[sys.argv[0]]+REST,verbosity=2)
