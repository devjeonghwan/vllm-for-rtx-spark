"""Retained-tail admission and physical-stride source contracts, CPU only.

Use --site for actual prepared vLLM source. The one-lane kernel interpreter
checks indexing/masks/lifetimes, not floating arithmetic or CUDA execution.
Real TopK16, FP32/64-noise and Graph tests belong to native component validation.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--site", type=Path)
ARGS, REST = parser.parse_known_args()
SITE = ARGS.site.resolve() if ARGS.site else None
ROOT = Path(__file__).resolve().parents[1]
REL = "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"


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


class TailManifest(unittest.TestCase):
    def test_tail_patch_follows_existing_state_fix_and_touches_only_speculator(self):
        names = json.loads((ROOT / "patches/vllm/manifest.json").read_text(encoding="utf-8"))["patches"]
        name = "0009-dflash2-retained-tail.patch"
        self.assertEqual(names.index(name), names.index("0008-mamba-resolved-state-unit.patch") + 1)
        text = (ROOT / "patches/vllm" / name).read_text(encoding="utf-8")
        self.assertEqual([line for line in text.splitlines() if line.startswith("+++ ")], ["+++ b/" + REL])


@unittest.skipUnless(SITE is not None, "Pass --site for prepared-source admission and kernel-stride contracts")
class TailContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((SITE / REL).read_text(encoding="utf-8"))
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
    unittest.main(argv=[sys.argv[0]] + REST, verbosity=2)
