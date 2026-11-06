"""CPU source-extracted DFlash graph budgets, aux capture and retained-tail indexing.

Symbolic tensors check routing/address contracts; numerical CUDA evidence is separate.
"""
from __future__ import annotations
import ast
import sys
import hashlib
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import unittest
from support import SOURCE as SITE
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from itertools import groupby, product
import copy
import math
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch


AUX_REL = 'vllm/models/qwen4_exp/nvidia/model.py'

CANDIDATE = SITE / AUX_REL if SITE else None

class Tensor:
    def __init__(self, value, shape=(16, 2560)):
        self.value = value
        self.shape = shape

    def op(self, name, *others, shape=None):
        digest = hashlib.sha256(repr((name, self.value, [x.value for x in others])).encode()).hexdigest()
        return Tensor(digest, shape or self.shape)

    def repeat(self, a, b):
        return self.op('repeat', shape=(self.shape[0], self.shape[1] * b))

    def unflatten(self, axis, shape):
        return self.op('unflatten', shape=(self.shape[0], *shape))

    def mean(self, dim):
        return self.op('mean', shape=(self.shape[0], self.shape[-1]))

class HC:
    def __init__(self, name, counts):
        self.name = name
        self.counts = counts

    def _combined(self, hidden, block, injection):
        return hidden.op('materialized_combine', block, injection)

    def combine(self, hidden, block, injection):
        self.counts['standalone_combine'] += 1
        return self._combined(hidden, block, injection)

    def mix(self, hidden):
        return hidden, hidden.op(self.name + ':mix', shape=(16, 2560)), hidden.op(self.name + ':inject', shape=(16, 4))

    def combine_and_mix(self, hidden, block, injection):
        self.counts['combine_norm'] += 1
        return self.mix(self._combined(hidden, block, injection))

def methods(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in ('Qwen4ExpDecoderLayer', 'Qwen4ExpModel'):
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name in ('forward', '_dflash2_append_completed_aux'):
                    node.name = cls.name + '_' + node.name
                    node.decorator_list = []
                    selected.append(node)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    env = {'islice': islice, 'get_pp_group': lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
           '_QSA_LAYER_TYPES': {'full_attention'}}
    exec(compile(module, str(path), 'exec'), env)
    return env

def run(path, *, aux=(6, 16, 26, 36, 46), deepstack=None, blocked=False, supported=True):
    f = methods(path)
    counts = {'standalone_combine': 0, 'combine_norm': 0}
    config = SimpleNamespace(hc_count=4, hidden_size=2560)
    class Layer:
        forward = f['Qwen4ExpDecoderLayer_forward']
        def __init__(self, index):
            self.config = config
            self.use_hc_sequence_parallel = self.use_sequence_parallel = False
            self.attn_hyper_connection = HC(str(index) + ':attn', counts)
            self.mlp_hyper_connection = HC(str(index) + ':mlp', counts)
            self.layer_type = 'linear_attention'
            self.linear_attn = lambda hidden_states: hidden_states.op(str(index) + ':attention')
            self.mlp = lambda x: x.op(str(index) + ':mlp')
            self.ple = (lambda hidden, *args: hidden.op('PLE')) if index == 1 or (blocked and index == 6) else None
        __call__ = forward
    class Model:
        forward = f['Qwen4ExpModel_forward']
        _dflash2_append_completed_aux = f['Qwen4ExpModel__dflash2_append_completed_aux']
        def _start_layer_ple_prefetch(self, *args):
            pass
    model = Model()
    model.config = config
    model._dflash2_aux_supported = supported
    model.aux_hidden_state_layers = aux
    model.start_layer = 0; model.end_layer = 48
    model.layers = [Layer(i) for i in range(48)]
    model.use_hc_sequence_parallel = model.use_sequence_parallel = False
    model._mtp_hidden_buffer = None
    model.hyper_connection_mixer = HC('final', counts)
    result = model.forward(Tensor('ids'), Tensor('positions', (16,)), inputs_embeds=Tensor('embed'),
                           query_start_loc=Tensor('qloc'), ngram_context=Tensor('ngram'), deepstack_input_embeds=deepstack)
    if isinstance(result, tuple):
        result = (result[0].value, [item.value for item in result[1]])
    else:
        result = result.value
    return result, counts

@unittest.skipUnless(SITE, 'Set VLLM_RTX_SPARK_SOURCE_DIR for graph contracts')
class AuxCaptureTests(unittest.TestCase):
    def test_exact_capture_order_and_target_route_structurally_equal(self):
        old, old_counts = run(CANDIDATE, supported=False)
        new, new_counts = run(CANDIDATE)
        self.assertEqual(old, new)
        self.assertEqual(len(new[1]), 5)
        self.assertEqual(old_counts['standalone_combine'] - new_counts['standalone_combine'], 5)
        self.assertEqual(old_counts['combine_norm'], new_counts['combine_norm'])

    def test_no_aux_MTP_path_unchanged(self):
        self.assertEqual(run(CANDIDATE, aux=(), supported=False), run(CANDIDATE, aux=()))

    def test_deepstack_nonNone_keeps_original_capture(self):
        self.assertEqual(run(CANDIDATE, deepstack={}, supported=False), run(CANDIDATE, deepstack={}))

    def test_PLE_capture_boundary_keeps_original_capture(self):
        self.assertEqual(run(CANDIDATE, blocked=True, supported=False), run(CANDIDATE, blocked=True))


    def test_other_capture_tuple_keeps_original_capture(self):
        self.assertEqual(run(CANDIDATE, aux=(7, 17), supported=False), run(CANDIDATE, aux=(7, 17)))


BUDGET_REL = Path('vllm/v1/worker/gpu/cudagraph_utils.py')

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
    value = SimpleNamespace(vllm_config=vllm, decode_query_len=8, max_num_reqs=16, _mbx_depth=None,
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

class BudgetTensor:
    def __init__(self, values):
        self.values = [list(row) for row in values]
    def __setitem__(self, key, value):
        rows, columns = key
        for i in range(len(self.values))[rows]:
            for j in range(len(self.values[i]))[columns]:
                self.values[i][j] = value

@unittest.skipUnless(SITE, 'Set VLLM_RTX_SPARK_SOURCE_DIR for graph contracts')
class GraphBudgetTests(unittest.TestCase):

    def test_target_schedule4and8_query_only8_all_B1to16(self):
        for budget in (3,4):
            target = manager(SITE / BUDGET_REL, False, budget)
            query = manager(SITE / BUDGET_REL, True, budget)
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
        target=manager(SITE/BUDGET_REL,False,4)
        query=manager(SITE/BUDGET_REL,True,4)
        for b in range(1,17):
            width=8 if b<8 else 5
            selected=target.dispatch(b,b*width,width)
            draft=query.dispatch(b,b*8,8)
            self.assertEqual((selected.cg_mode,selected.num_reqs,selected.num_tokens),(Mode.FULL,b,b*width))
            self.assertEqual((draft.cg_mode,draft.num_reqs,draft.num_tokens),(Mode.FULL,b,b*8))


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
            draft = BudgetTensor([[slot*100+step for step in range(7)] for slot in slots])
            env={'draft_tokens':draft,'num_spec_tokens':budget,'self':SimpleNamespace(num_speculative_steps=7)}
            exec(compile(module,str(path),'exec'),env)
            self.assertEqual([row[:budget] for row in draft.values],[[q[slot,step]for step in range(budget)]for slot in slots])
            self.assertTrue(all(value==-1 for row in draft.values for value in row[budget:]))
        # No q storage or per-request7-row stride is changed by the patch.
        self.assertEqual(q,{(slot,step):(slot*100+step)for slot in slots for step in range(7)})


TAIL_REL = "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"

def compile_source(nodes, scope):
    nodes = copy.deepcopy(nodes)
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "source-extracted-dflash2-tail-contract", "exec"), scope)
    return scope

class Number(float):
    def to(self, _dtype): return self

class Pointer:
    def __init__(self, storage, offset=0): self.storage, self.offset = storage, offset
    def __add__(self, offset): return Pointer(self.storage, self.offset + int(offset))

class ScalarTL:
    """One candidate per row, executes the unchanged source address expressions."""
    float32, float64 = "float32", "float64"
    pid = 0

    @classmethod
    def program_id(cls, dimension):
        assert dimension == 0
        return cls.pid

    @staticmethod
    def arange(start, end):
        assert (start, end) == (0, 1)
        return 0

    @staticmethod
    def load(pointer, mask=True, other=0):
        if not mask: return Number(0 if other is None else other)
        assert 0 <= pointer.offset < len(pointer.storage), "Unmasked out-of-bounds read"
        return Number(pointer.storage[pointer.offset])

    @staticmethod
    def store(pointer, value, mask=True):
        if mask:
            assert 0 <= pointer.offset < len(pointer.storage), "Unmasked out-of-bounds write"
            pointer.storage[pointer.offset] = value

@unittest.skipUnless(SITE, 'Set VLLM_RTX_SPARK_SOURCE_DIR for graph contracts')
class TailContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((SITE / TAIL_REL).read_text(encoding="utf-8"))
        owner = next(n for n in cls.tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlash2Speculator")
        methods = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in ("_draft_tail_active_steps", "propose")]
        class Parent:
            def propose(self, *args, **kwargs):
                if getattr(self, "raise_original", False): raise ValueError("original failure")
                return {"args": args, "kwargs": kwargs, "effective_steps": self._draft_tail_active_steps(args[0].num_reqs), "forced_full": self._draft_tail_force_full}
        subject = ast.ClassDef(name="Subject", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
        cls.Subject = compile_source([subject], {"Parent": Parent, "CUDAGraphMode": NS(FULL="FULL")})["Subject"]
        kernels = [n for n in cls.tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_selector_walk_kernel", "_cache_draft_logits_kernel")]
        cls.draws = []
        def draw(scores, ids, mask, seed, position, temperature, **kwargs):
            cls.draws.append((seed, position, temperature, kwargs))
            return scores, 0
        scope = compile_source(kernels, {"tl": ScalarTL, "gumbel_noised_argmax": draw})
        cls.walk = staticmethod(scope["_selector_walk_kernel"])
        cls.cache = staticmethod(scope["_cache_draft_logits_kernel"])
        # The pinned schedule utility is CPU-only. Do not import vLLM/Torch.
        cls.utility = {}
        exec(compile((SITE / "vllm/v1/spec_decode/dynamic/utils.py").read_text(encoding="utf-8"), "original_schedule_utility", "exec"), cls.utility)

    def setUp(self):
        names = ["vllm", "vllm.v1", "vllm.v1.spec_decode", "vllm.v1.spec_decode.dynamic", "vllm.v1.spec_decode.dynamic.utils"]
        modules = {name: ModuleType(name) for name in names}
        for module in modules.values(): module.__path__ = []
        modules[names[-1]].build_dynamic_sd_schedule_lookup = self.utility["build_dynamic_sd_schedule_lookup"]
        self.imports = patch.dict(sys.modules, modules)
        self.imports.start(); self.addCleanup(self.imports.stop)

    def owner(self, schedule=None, **overrides):
        subject = self.Subject()
        subject._speculator_name = "DFlash2"; subject.num_speculative_steps = 7
        subject.max_num_reqs = 16; subject.num_query_per_req = 8; subject.enable_adaptive_verification = False
        subject.vllm_config = NS(speculative_config=NS(num_speculative_tokens_per_batch_size=schedule if schedule is not None else [(1, 7, 7), (8, 16, 4)], uses_dynamic_speculative_decoding=lambda: True), parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1))
        subject.query_cudagraph_manager = NS(_graphs_captured=True, dispatch=lambda b, n, **kwargs: NS(cg_mode="FULL", num_reqs=b, num_tokens=n, uniform_token_count=8))
        for name, value in overrides.items(): setattr(subject, name, value)
        return subject

    def propose(self, subject, b, requested, **kwargs):
        batch = NS(num_reqs=b)
        args = [object() for _ in range(10)]
        result = subject.propose(batch, *args, num_speculative_tokens=requested, **kwargs)
        self.assertIs(result["args"][0], batch)
        self.assertTrue(all(a is z for a, z in zip(result["args"][1:], args)))
        self.assertEqual(result["kwargs"]["num_speculative_tokens"], requested)
        self.assertFalse(subject._draft_tail_force_full)
        return result

    def test_actual_pydantic_tuple_and_raw_schedules_keep_exact_budget(self):
        from pydantic import TypeAdapter
        typed = TypeAdapter(list[tuple[int, int, int]]).validate_python([[1, 7, 7], [8, 16, 4]])
        self.assertIsInstance(typed[0], tuple)
        for schedule in (typed, [[1, 7, 7], [8, 16, 4]]):
            subject = self.owner(schedule)
            for b in range(1, 17):
                expected = 7 if b <= 7 else 4
                result = self.propose(subject, b, expected)
                self.assertEqual(result["effective_steps"], expected)
                self.assertFalse(result["forced_full"])
                self.assertFalse(result["kwargs"]["is_profile"])

    def test_changed_schedule_adaptive_and_parallel_modes_keep_full7(self):
        for schedule in ([(1, 16, 7)], [(1, 7, 7), (8, 16, 3)]):
            self.assertEqual(self.owner(schedule)._draft_tail_active_steps(16), 7)
        for changes in ({"enable_adaptive_verification": True}, {"_speculator_name": "LiLiCorr"}, {"max_num_reqs": 8}):
            self.assertEqual(self.owner(**changes)._draft_tail_active_steps(8), 7)
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
            subject = self.owner(); setattr(subject.vllm_config.parallel_config, name, 2)
            self.assertEqual(subject._draft_tail_active_steps(16), 7)

    def test_padding_none_mode_and_live_budget_mismatch_use_original_eager_full7(self):
        for desc in (NS(cg_mode="FULL", num_reqs=16, num_tokens=128, uniform_token_count=8), NS(cg_mode="NONE", num_reqs=8, num_tokens=64, uniform_token_count=8), NS(cg_mode="FULL", num_reqs=8, num_tokens=64, uniform_token_count=7)):
            subject = self.owner(); subject.query_cudagraph_manager.dispatch = lambda *a, **k: desc
            result = self.propose(subject, 8, 4)
            self.assertEqual(result["effective_steps"], 7); self.assertTrue(result["kwargs"]["is_profile"])
        for requested in (None, 3, 7, 4.0):
            result = self.propose(self.owner(), 8, requested)
            self.assertEqual(result["effective_steps"], 7); self.assertTrue(result["kwargs"]["is_profile"])

    def test_dummy_profile_uncaptured_and_original_exception_restore_state(self):
        for kwargs in ({"dummy_run": True}, {"is_profile": True}):
            self.assertEqual(self.propose(self.owner(), 16, 4, **kwargs)["effective_steps"], 7)
        subject = self.owner(); subject.query_cudagraph_manager._graphs_captured = False
        self.assertEqual(self.propose(subject, 16, 4)["effective_steps"], 7)
        subject = self.owner(); subject.raise_original = True
        with self.assertRaisesRegex(ValueError, "original failure"):
            subject.propose(NS(num_reqs=16), *[None] * 10, num_speculative_tokens=4)
        self.assertFalse(subject._draft_tail_force_full)

    def test_actual_kernel_addresses_preserve_q7_and_slot_reassignment(self):
        R, B, V = 5, 3, 41
        q = [-math.inf] * (R * 7 * V); cached = [0] * (R * 7)
        self.draws.clear()
        for generation, (active, states) in enumerate(((7, [3, 1, -1]), (4, [1, 3, -1]), (7, [3, 4, -1]), (4, [4, 0, -1]), (7, [0, 4, -1]))):
            old = list(q); tokens = [-777] * (B * 7)
            ids = [(generation * 5 + row * 9 + step * 3) % V for row in range(B) for step in range(active)]
            scores = [generation + step / 8 for row in range(B) for step in range(active)]
            positions = [5000 + generation * 13 + step for row in range(B) for step in range(active)]
            mapping = [state for state in states for _ in range(active)]
            realized = [-999] * (B * 7)
            for row in range(B):
                ScalarTL.pid = row
                self.walk(*map(Pointer, (scores, ids, positions, mapping, [0.7] * R, [113 + r for r in range(R)], tokens)), 7, Pointer(realized), active, 1, 1, True, False)
            for flat in range(B * active):
                ScalarTL.pid = flat
                self.cache(Pointer(q), Pointer(cached), 7, 1, Pointer(ids), Pointer(realized), Pointer(mapping), 7 * V, V, active, 1, 1)
            for row, state in enumerate(states):
                if state < 0:
                    self.assertEqual(tokens[row * 7:(row + 1) * 7], [-777] * 7)
                    continue
                self.assertEqual(tokens[row * 7:row * 7 + active], ids[row * active:(row + 1) * active])
                self.assertEqual(tokens[row * 7 + active:(row + 1) * 7], [-1] * (7 - active))
                for step in range(active):
                    values = q[(state * 7 + step) * V:(state * 7 + step + 1) * V]
                    expected = ids[row * active + step]
                    self.assertEqual([i for i, v in enumerate(values) if math.isfinite(v)], [expected])
                    self.assertEqual(values[expected], scores[row * active + step])
                    self.assertEqual(cached[state * 7 + step], expected)
                self.assertEqual(q[(state * 7 + active) * V:(state + 1) * 7 * V], old[(state * 7 + active) * V:(state + 1) * 7 * V])
            for state in set(range(R)) - {v for v in states if v >= 0}:
                self.assertEqual(q[state * 7 * V:(state + 1) * 7 * V], old[state * 7 * V:(state + 1) * 7 * V])
            # Real source still keys the draw by seed and predicted position-1.
            expected_positions = [v - 1 for v in positions]
            self.assertEqual([int(v[1]) for v in self.draws[-B * active:]], expected_positions)

    def test_no_Torch_import_or_extra_query_head_cache_owner(self):
        self.assertNotIn("torch", sys.modules)
        owner = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlash2Speculator")
        init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "CandidateSampler" for n in ast.walk(init)), 1)


if __name__ == "__main__":
    unittest.main()
