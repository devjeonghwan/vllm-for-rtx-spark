"""CPU-only contracts for the optional DFlash2 projection fork and join.

Import the actual helper with a fake Torch module; no Torch, CUDA or vLLM
runtime is imported. Optional --site/--baseline inspect composed source.
"""
import argparse
import ast
import builtins
from copy import deepcopy
import importlib.util
from pathlib import Path
import runpy
import sys
from types import MethodType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--site", type=Path)
parser.add_argument("--baseline", type=Path)
ARGS, REST = parser.parse_known_args()
PROJECT = ARGS.project.resolve()
SITE = ARGS.site.resolve() if ARGS.site else None
BASE = ARGS.baseline.resolve() if ARGS.baseline else None
HELPER = PROJECT / "src/recipes/qwen3_8/dflash_head_overlap.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("source_only_dflash_head_overlap", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forbid_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("Disabled overlap must not import Torch")
    return ORIGINAL_IMPORT(name, *args, **kwargs)


ORIGINAL_IMPORT = builtins.__import__


def config(additional=None):
    return NS(additional_config={} if additional is None else additional,
        speculative_config=NS(method="dflash", draft_model_config=NS(hf_config=NS(
            architectures=["DFlash2DraftModel"]))),
        model_config=NS(hf_config=NS(architectures=["Qwen4ExpForConditionalGeneration"])),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
            data_parallel_size=1, prefill_context_parallel_size=1,
            decode_context_parallel_size=1))


class FakeStream:
    def __init__(self, cuda, name):
        self.cuda, self.name = cuda, name

    def wait_event(self, event):
        self.cuda.log.append(("wait", self.name, event.name))
        if self.cuda.fail_wait == self.name:
            raise RuntimeError("injected stream join failure")


class FakeEvent:
    def __init__(self, cuda, name):
        self.cuda, self.name = cuda, name

    def record(self, stream):
        self.cuda.log.append(("record", self.name, stream.name))


class StreamContext:
    def __init__(self, cuda, stream):
        self.cuda, self.stream = cuda, stream

    def __enter__(self):
        self.cuda.log.append(("enter", self.stream.name))
        self.previous = self.cuda.active
        self.cuda.active = self.stream

    def __exit__(self, *error):
        self.cuda.log.append(("exit", self.stream.name))
        self.cuda.active = self.previous


class FakeCUDA:
    def __init__(self):
        self.log, self.streams, self.events = [], [], []
        self.fail_wait = None
        self.main = FakeStream(self, "main")
        self.active = self.main

    def Stream(self, *, device):
        stream = FakeStream(self, "side" + str(len(self.streams)))
        self.streams.append(stream)
        return stream

    def Event(self):
        event = FakeEvent(self, "ready" if len(self.events) % 2 == 0 else "done")
        self.events.append(event)
        return event

    def current_stream(self, device):
        return self.main

    def stream(self, stream):
        return StreamContext(self, stream)


class FakeTensor:
    def __init__(self, cuda, name, shape=()):
        self.cuda, self.name, self.shape = cuda, name, shape
        self.device = NS(type="cuda")
        self.recorded = []

    def record_stream(self, stream):
        self.recorded.append(stream)
        self.cuda.log.append(("lifetime", self.name, stream.name))

    # Shape-only objects for executing the actual _generate_draft control flow.
    # They do not imitate numerical indexing, projection or sampling.
    def __getitem__(self, key):return self
    def view(self, *shape):return self
    def reshape(self, *shape):return self
    def view_as(self, other):return self
    def flatten(self, *dimensions):return self


class InputIDs:
    def __init__(self):
        self.slices = []
        self.anchors = object()

    def __getitem__(self, key):
        self.slices.append(key)
        return self.anchors


class CandidateSelector:
    """Only Python call/owner contracts; no numerical selector implementation."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.predecessor_codebook, self.successor_codebook = object(), object()
        self.top_k = 16
        self.hidden_projection = fixture.project
        self._forward_pre_hooks, self._forward_hooks = {}, {}
        self._compiled_call_impl = None

    def forward(self, *args):
        self.fixture.parent_calls.append(args)
        return self.fixture.parent_result

    def __call__(self, *args):
        self.fixture.cuda.log.append(("parent_selector",))
        return self.forward(*args)


class RuntimeFixture:
    def __init__(self):
        self.cuda = FakeCUDA()
        self.torch = ModuleType("torch")
        self.torch.cuda = self.cuda
        self.global_hooks = NS(_global_forward_pre_hooks={}, _global_forward_hooks={})
        self.torch.nn = NS(modules=NS(module=self.global_hooks))
        self.score_calls = []
        self.parent_calls = []
        self.parent_result = object()
        self.score_error = None
        self.result = object()
        leaf = ModuleType("vllm.model_executor.models.qwen3_dflash2")
        leaf._score_edges = self.score
        leaf.CandidateSelector = CandidateSelector
        self.modules = {"torch": self.torch, leaf.__name__: leaf}
        for name in ("vllm", "vllm.model_executor", "vllm.model_executor.models"):
            package = ModuleType(name)
            package.__path__ = []
            self.modules[name] = package
        self.inputs = InputIDs()
        self.selector = CandidateSelector(self)
        self.speculator = NS(model=NS(model=NS(candidate_selector=self.selector)),
            input_buffers=NS(input_ids=self.inputs), num_query_per_req=8)
        self.ids = FakeTensor(self.cuda, "ids", (16, 4, 16))
        self.unary = FakeTensor(self.cuda, "unary", (16, 4, 16))
        self.hidden = FakeTensor(self.cuda, "hidden", (16, 4, 2560))
        self.projected = FakeTensor(self.cuda, "projected", (16, 4, 256))
        self.projection_error = None
        self.project_calls = []

    def project(self, hidden):
        self.project_calls.append(hidden)
        self.cuda.log.append(("projection", self.cuda.active.name))
        if self.projection_error:
            raise self.projection_error
        return self.projected

    def score(self, *args):
        self.score_calls.append(args)
        self.cuda.log.append(("score",))
        if self.score_error:
            raise self.score_error
        return self.result


class OptionContracts(unittest.TestCase):
    def test_actual_recipe_resolves_and_validates_the_opt_in_without_cuda(self):
        with patch("builtins.__import__", forbid_torch):
            runpy.run_path(PROJECT / "scripts/load_source_package.py")["load"]()
            from _rtx_spark.recipes import qwen3_8 as recipe

            self.assertIs(recipe._validate_options({})["dflash2_head_overlap"], False)
            nested = {"optimizations": {"dflash2_head_overlap": True}}
            self.assertIs(recipe._validate_options(nested)["dflash2_head_overlap"], True)
            # Omitted/default OFF must validate even without CUDA/model fields.
            recipe.validate(object(), {})
            recipe.validate(config(), nested)
            for value in (0, 1, "true", None):
                with self.subTest(value=value), self.assertRaises(TypeError):
                    recipe._validate_options({"dflash2_head_overlap": value})
            with self.assertRaises(ValueError):
                recipe._validate_options({"dflash2_head_overlap": False, **nested})
            mtp = config()
            mtp.speculative_config.method = "mtp"
            recipe.validate(mtp, {})
            with self.assertRaisesRegex(ValueError, "DFlash2 draft"):
                recipe.validate(mtp, nested)

    def test_helper_is_in_the_actual_builder_runtime_payload(self):
        # prepare intentionally has no _rtx_spark directory. The ordinary
        # builder collects src under that namespace later; collect only, never
        # invoke build/wheel or read native assets.
        builder = runpy.run_path(PROJECT / "scripts/build_vllm.py")
        payload = {relative: path for path, relative in builder["payload_files"](PROJECT / "src")}
        self.assertEqual(payload["recipes/qwen3_8/dflash_head_overlap.py"].resolve(), HELPER.resolve())
        self.assertEqual(payload["recipes/qwen3_8/dflash_head_overlap.py"].read_bytes(), HELPER.read_bytes())

    def test_import_omitted_false_and_other_recipe_do_not_import_torch(self):
        with patch("builtins.__import__", forbid_torch):
            helper = load_helper()
            helper.validate_config(object(), False)
            for options in (None, {}, {"rtx_spark": {}}, {"rtx_spark": "other"},
                    {"rtx_spark": {"model": "unrelated"}},
                    {"rtx_spark": {"dflash2_head_overlap": False}},
                    {"rtx_spark": {"optimizations": {"dflash2_head_overlap": False}}}):
                # Deliberately omit all architecture/parallel fields: OFF must
                # return before validation or CUDA owner construction.
                self.assertIsNone(helper.create_overlap(NS(additional_config=options), NS(type="cuda")))

    def test_flat_nested_and_matching_repeated_true_create_one_owner(self):
        helper = load_helper()
        for opts in ({"dflash2_head_overlap": True},
                {"optimizations": {"dflash2_head_overlap": True}},
                {"dflash2_head_overlap": True, "optimizations": {"dflash2_head_overlap": True}}):
            f = RuntimeFixture()
            with patch.dict(sys.modules, f.modules):
                owner = helper.create_overlap(config({"rtx_spark": opts}), NS(type="cuda"))
            self.assertIsInstance(owner, helper.SelectorProjectionOverlap)
            self.assertIs(owner.stream, f.cuda.streams[0])
            self.assertEqual((len(f.cuda.streams), len(f.cuda.events)), (1, 2))
            self.assertFalse(owner.pending)
            self.assertIsNone(owner.hidden)
            self.assertIsNone(owner.projected)

    def test_strict_bool_and_conflict_before_any_torch_import(self):
        helper = load_helper()
        with patch("builtins.__import__", forbid_torch):
            for invalid in (0, 1, "true", None, [], {}):
                for opts in ({"dflash2_head_overlap": invalid},
                        {"optimizations": {"dflash2_head_overlap": invalid}},
                        {"dflash2_head_overlap": False, "optimizations": {"dflash2_head_overlap": invalid}}):
                    with self.subTest(invalid=invalid, opts=opts), self.assertRaises(TypeError):
                        helper.create_overlap(NS(additional_config={"rtx_spark": opts}), NS(type="cuda"))
            for flat in (False, True):
                with self.assertRaisesRegex(ValueError, "Conflicting"):
                    helper.create_overlap(NS(additional_config={"rtx_spark": {
                        "dflash2_head_overlap": flat,
                        "optimizations": {"dflash2_head_overlap": not flat}}}), NS(type="cuda"))

    def test_explicit_enable_rejects_mtp_wrong_architecture_and_parallelism(self):
        helper = load_helper()
        with patch("builtins.__import__", forbid_torch):
            for method in ("mtp", "eagle", "dspark", None):
                c = config({"rtx_spark": {"dflash2_head_overlap": True}})
                c.speculative_config.method = method
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "DFlash2 draft"):
                    helper.create_overlap(c, NS(type="cuda"))
            for path in ("draft", "target"):
                c = config({"rtx_spark": {"dflash2_head_overlap": True}})
                if path == "draft":c.speculative_config.draft_model_config.hf_config.architectures = ["DFlashDraftModel"]
                else:c.model_config.hf_config.architectures = ["Qwen3ForCausalLM"]
                with self.assertRaises(ValueError):helper.create_overlap(c, NS(type="cuda"))
            for name in vars(config().parallel_config):
                c = config({"rtx_spark": {"dflash2_head_overlap": True}})
                setattr(c.parallel_config, name, 2)
                with self.subTest(parallel=name), self.assertRaisesRegex(ValueError, "TP1/PP1/DP1/CP1"):
                    helper.create_overlap(c, NS(type="cuda"))

    def test_cpu_device_rejected_before_stream_allocation(self):
        f = RuntimeFixture()
        with patch.dict(sys.modules, f.modules), self.assertRaisesRegex(ValueError, "CUDA device"):
            load_helper().create_overlap(config({"rtx_spark": {"dflash2_head_overlap": True}}), NS(type="cpu"))
        self.assertEqual((f.cuda.streams, f.cuda.events), ([], []))


class LifetimeContracts(unittest.TestCase):
    def setUp(self):
        self.helper = load_helper()
        self.f = RuntimeFixture()
        context = patch.dict(sys.modules, self.f.modules)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.owner = self.helper.SelectorProjectionOverlap(NS(type="cuda"))

    def assert_clean(self):
        self.assertFalse(self.owner.pending)
        self.assertIsNone(self.owner.hidden)
        self.assertIsNone(self.owner.projected)
        self.assertIsNone(self.owner.selector)
        self.assertIsNone(self.owner.projector)

    def test_original_projection_score_arguments_dependency_order_and_reuse(self):
        f, owner = self.f, self.owner
        stream, ready, done = owner.stream, owner.ready, owner.done
        for repeat in range(2):
            f.cuda.log.clear()
            self.assertTrue(owner.start(f.hidden, f.selector))
            # The caller submits the unchanged full head and TopK here.
            f.cuda.log.append(("head_and_topk",))
            actual = owner.score(f.speculator, f.ids, f.unary)
            self.assertIs(actual, f.result)
            self.assertEqual(f.cuda.log, [
                ("record", "ready", "main"), ("wait", "side0", "ready"),
                ("lifetime", "hidden", "side0"), ("enter", "side0"),
                ("projection", "side0"), ("record", "done", "side0"), ("exit", "side0"),
                ("head_and_topk",), ("wait", "main", "done"),
                ("lifetime", "projected", "main"), ("score",)])
            self.assertIs(f.project_calls[-1], f.hidden)
            args = f.score_calls[-1]
            self.assertEqual(len(args), 7)
            for got, expected in zip(args[:6], (f.selector.predecessor_codebook, f.selector.successor_codebook,
                    f.ids, f.unary, f.projected, f.inputs.anchors)):
                self.assertIs(got, expected)
            self.assertEqual(args[-1], 16)
            self.assertEqual(f.inputs.slices[-1], slice(None, 16 * 8, 8))
            self.assert_clean()
            self.assertIs(owner.stream, stream)
            self.assertIs(owner.ready, ready)
            self.assertIs(owner.done, done)
        self.assertEqual(len(f.score_calls), 2)
        self.assertEqual((len(f.cuda.streams), len(f.cuda.events)), (1, 2))

    def test_projection_failure_retains_hidden_owner_and_blocks_next_start(self):
        f, owner = self.f, self.owner
        f.projection_error = RuntimeError("projection failure after possible submission")
        with self.assertRaisesRegex(RuntimeError, "projection failure"):
            owner.start(f.hidden, f.selector)
        self.assertTrue(owner.pending)
        self.assertIs(owner.hidden, f.hidden)
        self.assertIsNone(owner.projected)
        self.assertEqual(f.hidden.recorded, [owner.stream])
        with self.assertRaisesRegex(RuntimeError, "Unjoined"):
            owner.start(f.hidden, f.selector)
        with self.assertRaisesRegex(RuntimeError, "not completed submission"):
            owner.score(f.speculator, f.ids, f.unary)
        self.assertEqual(len(f.project_calls), 1)
        self.assertEqual(f.score_calls, [])

    def test_head_failure_and_reentrant_start_keep_projection_until_join(self):
        f, owner = self.f, self.owner
        self.assertTrue(owner.start(f.hidden, f.selector))
        try:
            raise RuntimeError("caller head failure")
        except RuntimeError:
            pass
        self.assertTrue(owner.pending)
        self.assertIs(owner.hidden, f.hidden)
        self.assertIs(owner.projected, f.projected)
        with self.assertRaisesRegex(RuntimeError, "Unjoined"):
            owner.start(f.hidden, f.selector)
        self.assertEqual(len(f.project_calls), 1)
        self.assertEqual(f.score_calls, [])

    def test_join_failure_keeps_strong_owners_and_no_score(self):
        f, owner = self.f, self.owner
        self.assertTrue(owner.start(f.hidden, f.selector))
        f.cuda.fail_wait = "main"
        with self.assertRaisesRegex(RuntimeError, "join failure"):
            owner.score(f.speculator, f.ids, f.unary)
        self.assertTrue(owner.pending)
        self.assertIs(owner.hidden, f.hidden)
        self.assertIs(owner.projected, f.projected)
        self.assertEqual(f.projected.recorded, [])
        self.assertEqual(f.score_calls, [])
        with self.assertRaisesRegex(RuntimeError, "Unjoined"):
            owner.start(f.hidden, f.selector)

    def test_score_failure_after_join_clears_only_lifetime_tracked_references(self):
        f, owner = self.f, self.owner
        self.assertTrue(owner.start(f.hidden, f.selector))
        f.score_error = RuntimeError("original score failure")
        with self.assertRaisesRegex(RuntimeError, "original score failure"):
            owner.score(f.speculator, f.ids, f.unary)
        self.assertEqual(f.hidden.recorded, [owner.stream])
        self.assertEqual(f.projected.recorded, [f.cuda.main])
        self.assertEqual(len(f.score_calls), 1)
        self.assert_clean()
        f.score_error = None
        self.assertTrue(owner.start(f.hidden, f.selector))
        self.assertIs(owner.score(f.speculator, f.ids, f.unary), f.result)
        self.assert_clean()

    def test_score_without_start_rejected_without_cuda_work(self):
        with self.assertRaisesRegex(RuntimeError, "not completed submission"):
            self.owner.score(self.f.speculator, self.f.ids, self.f.unary)
        self.assertEqual(self.f.cuda.log, [])
        self.assertEqual(self.f.score_calls, [])

    def test_parent_local_global_hooks_or_compilation_keep_original_path(self):
        f, owner = self.f, self.owner
        tables = (f.selector._forward_pre_hooks, f.selector._forward_hooks,
            f.global_hooks._global_forward_pre_hooks, f.global_hooks._global_forward_hooks)
        for table in tables:
            with self.subTest(table=id(table)):
                table[1] = object()
                self.assertFalse(owner.start(f.hidden, f.selector))
                self.assert_clean()
                self.assertEqual(f.cuda.log, [])
                self.assertEqual(f.project_calls, [])
                table.clear()
        f.selector._compiled_call_impl = object()
        self.assertFalse(owner.start(f.hidden, f.selector))
        self.assert_clean()
        self.assertEqual(f.cuda.log, [])

    def test_subclass_and_forward_or_call_overrides_are_not_bypassed(self):
        f, owner = self.f, self.owner
        subclass = type("CustomSelector", (CandidateSelector,), {})
        self.assertFalse(owner.start(f.hidden, subclass(f)))
        def custom_forward(selector, *args):return object()
        with patch.object(f.selector, "forward", MethodType(custom_forward, f.selector)):
            self.assertFalse(owner.start(f.hidden, f.selector))
        with patch.object(CandidateSelector, "forward", custom_forward):
            self.assertFalse(owner.start(f.hidden, f.selector))
        with patch.object(CandidateSelector, "__call__", custom_forward):
            self.assertFalse(owner.start(f.hidden, f.selector))
        self.assert_clean()
        self.assertEqual(f.cuda.log, [])
        self.assertEqual(f.project_calls, [])

    def test_hook_registered_after_projection_joins_then_calls_original_selector(self):
        f, owner = self.f, self.owner
        self.assertTrue(owner.start(f.hidden, f.selector))
        f.selector._forward_hooks[1] = object()
        actual = owner.score(f.speculator, f.ids, f.unary)
        self.assertIs(actual, f.parent_result)
        self.assertEqual(f.score_calls, [])
        args = f.parent_calls[0]
        for got, expected in zip(args, (f.ids, f.unary, f.hidden, f.inputs.anchors)):
            self.assertIs(got, expected)
        self.assertLess(f.cuda.log.index(("wait", "main", "done")),
                        f.cuda.log.index(("parent_selector",)))
        self.assertEqual(f.projected.recorded, [f.cuda.main])
        self.assert_clean()

    def test_pending_guard_precedes_newly_unsupported_fallback(self):
        f, owner = self.f, self.owner
        self.assertTrue(owner.start(f.hidden, f.selector))
        f.selector._forward_hooks[1] = object()
        with self.assertRaisesRegex(RuntimeError, "Unjoined"):
            owner.start(f.hidden, f.selector)
        self.assertTrue(owner.pending)
        self.assertIs(owner.hidden, f.hidden)
        self.assertIs(owner.projected, f.projected)

    def test_selector_replaced_during_head_uses_new_original_selector_after_join(self):
        f, owner = self.f, self.owner
        original = f.selector
        self.assertTrue(owner.start(f.hidden, original))
        replacement = CandidateSelector(f)
        f.speculator.model.model.candidate_selector = replacement
        self.assertIs(owner.score(f.speculator, f.ids, f.unary), f.parent_result)
        self.assertEqual(f.score_calls, [])
        self.assertEqual(len(f.parent_calls), 1)
        self.assertIs(f.parent_calls[0][2], f.hidden)
        self.assertLess(f.cuda.log.index(("wait", "main", "done")),
                        f.cuda.log.index(("parent_selector",)))
        self.assert_clean()

    def test_projector_replaced_during_head_is_not_mixed_with_old_projection(self):
        f, owner = self.f, self.owner
        original_projector = f.selector.hidden_projection
        self.assertTrue(owner.start(f.hidden, f.selector))
        self.assertIs(owner.projector, original_projector)
        f.selector.hidden_projection = lambda hidden: object()
        self.assertIs(owner.score(f.speculator, f.ids, f.unary), f.parent_result)
        self.assertEqual(f.score_calls, [])
        self.assertEqual(len(f.project_calls), 1)
        self.assertEqual(len(f.parent_calls), 1)
        self.assertEqual(f.projected.recorded, [f.cuda.main])
        self.assert_clean()


def tree(path):return ast.parse(path.read_text(encoding="utf-8"))
def dump(node):return ast.dump(node, include_attributes=False)
def function(module, name, cls=None):
    body = module.body if cls is None else next(n.body for n in module.body
        if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def replace_exact_sequence(body, before, after):
    """One exact sanctioned span; never remove arbitrary statements/branches."""
    hits = [i for i in range(len(body) - len(before) + 1)
            if all(dump(a) == dump(b) for a, b in zip(body[i:i + len(before)], before))]
    if len(hits) != 1:
        raise AssertionError("Expected one exact overlap source boundary")
    start = hits[0]
    body[start:start + len(before)] = deepcopy(after)


@unittest.skipUnless(SITE and BASE, "Pass --site and --baseline for composed-source contracts")
class ComposedSourceContracts(unittest.TestCase):
    REL = "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"

    def test_exact_optional_diff_reversed_restores_entire_original_module(self):
        old, new = tree(BASE / self.REL), tree(SITE / self.REL)
        init = function(new, "__init__", "DFlash2Speculator")
        extra_init = ast.parse("""from vllm._rtx_spark.recipes.qwen3_8.dflash_head_overlap import create_overlap
self._selector_projection_overlap = create_overlap(vllm_config, device)
""").body
        replace_exact_sequence(init.body, extra_init, [])
        generate = function(new, "_generate_draft", "DFlash2Speculator")
        prefix = ast.parse("""overlap = self._selector_projection_overlap
overlap_started = False
if overlap is not None:
    overlap_started = overlap.start(hidden_states, self.model.model.candidate_selector)
""").body
        replace_exact_sequence(generate.body, prefix, [])
        branch = ast.parse("""if not overlap_started:
    scores = self._score_candidates(candidate_ids, unary_logits, hidden_states)
else:
    scores = overlap.score(self, candidate_ids, unary_logits)
""").body
        original = ast.parse("scores = self._score_candidates(candidate_ids, unary_logits, hidden_states)").body
        replace_exact_sequence(generate.body, branch, original)
        self.assertEqual(dump(new), dump(old))

    def test_sampler_q_rng_target_verifier_and_model_math_unchanged(self):
        old, new = tree(BASE / self.REL), tree(SITE / self.REL)
        old_sampler = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "CandidateSampler")
        new_sampler = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "CandidateSampler")
        self.assertEqual(dump(new_sampler), dump(old_sampler))
        for name in ("_selector_walk_kernel", "_cache_draft_logits_kernel"):
            self.assertEqual(dump(function(new, name)), dump(function(old, name)))
        for relative in ("vllm/model_executor/models/qwen3_dflash2.py",
                "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py",
                "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
                "vllm/v1/worker/gpu/sample/gumbel.py"):
            with self.subTest(path=relative):
                self.assertEqual((SITE / relative).read_bytes(), (BASE / relative).read_bytes())
        # Project helpers are packaged later, not copied into prepare output.
        for relative in ("runtime/sampling.py", "kernels/sampling/rejection.py", "kernels/dflash2/candidate_edges.py"):
            with self.subTest(helper=relative):
                self.assertEqual((PROJECT / "src" / relative).read_bytes(),
                                 (BASE / "vllm/_rtx_spark" / relative).read_bytes())

    def generate_fixture(self, overlap):
        f = RuntimeFixture()
        production = tree(SITE / self.REL)
        methods = [deepcopy(function(production, name, "DFlash2Speculator"))
                   for name in ("_generate_draft", "_score_candidates")]
        # Future annotations avoid importing Torch/vLLM solely for type names.
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        scope = {"CUDAGraphMode": NS(NONE=0)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])),
            "actual-composed-DFlash2-control-flow", "exec"), scope)
        spec = f.speculator
        spec._selector_projection_overlap = overlap
        spec.num_speculative_steps, spec.top_k = 7, 16
        spec._draft_tail_active_steps = lambda count: 4
        spec._run_model = lambda *args: f.hidden
        spec.sample_indices = FakeTensor(f.cuda, "indices")
        for name in ("sample_pos", "sample_idx_mapping", "temperature", "seeds", "draft_tokens", "draft_logits"):
            setattr(spec, name, FakeTensor(f.cuda, name))
        spec.use_fp64_gumbel, spec.enable_adaptive_verification = False, False
        f.sample_calls = []
        spec.candidate_sampler = NS(sample=lambda *args, **kwargs: f.sample_calls.append((args, kwargs)))
        spec._score_candidates = MethodType(scope["_score_candidates"], spec)
        def head(hidden):
            f.cuda.log.append(("head_and_topk",))
            if getattr(f, "head_error", None):raise f.head_error
            if getattr(f, "head_action", None):f.head_action()
            return f.ids, f.unary
        spec.model.compute_candidates = head
        return f, MethodType(scope["_generate_draft"], spec)

    def test_actual_generate_disabled_uses_original_selector_and_sampler_once(self):
        f, generate = self.generate_fixture(None)
        with patch.dict(sys.modules, f.modules):generate(16, 128, None, None, None)
        self.assertEqual((len(f.parent_calls), len(f.sample_calls)), (1, 1))
        self.assertEqual(f.score_calls, [])
        self.assertEqual((f.cuda.streams, f.cuda.events), ([], []))
        self.assertIs(f.sample_calls[0][0][1], f.parent_result)

    def test_actual_generate_enabled_fork_head_join_then_original_sampler(self):
        f, generate = self.generate_fixture(None)
        with patch.dict(sys.modules, f.modules):
            owner = load_helper().SelectorProjectionOverlap(NS(type="cuda"))
            f.speculator._selector_projection_overlap = owner
            generate(16, 128, None, None, None)
        self.assertEqual((len(f.project_calls), len(f.score_calls), len(f.sample_calls)), (1, 1, 1))
        self.assertIs(f.sample_calls[0][0][1], f.result)
        self.assertEqual(f.parent_calls, [])
        self.assertLess(f.cuda.log.index(("projection", "side0")), f.cuda.log.index(("head_and_topk",)))
        self.assertLess(f.cuda.log.index(("head_and_topk",)), f.cuda.log.index(("wait", "main", "done")))
        self.assertFalse(owner.pending)

    def test_actual_generate_parent_hook_falls_back_and_head_failure_keeps_pending(self):
        for fail_head in (False, True):
            f, generate = self.generate_fixture(None)
            with patch.dict(sys.modules, f.modules):
                owner = load_helper().SelectorProjectionOverlap(NS(type="cuda"))
                f.speculator._selector_projection_overlap = owner
                if fail_head:
                    f.head_error = RuntimeError("actual head failure")
                    with self.assertRaisesRegex(RuntimeError, "actual head failure"):
                        generate(16, 128, None, None, None)
                    self.assertTrue(owner.pending)
                    self.assertIs(owner.hidden, f.hidden)
                    self.assertIs(owner.projected, f.projected)
                    self.assertEqual(f.sample_calls, [])
                    with self.assertRaisesRegex(RuntimeError, "Unjoined"):
                        owner.start(f.hidden, f.selector)
                else:
                    f.selector._forward_pre_hooks[1] = object()
                    generate(16, 128, None, None, None)
                    self.assertEqual((len(f.parent_calls), len(f.sample_calls)), (1, 1))
                    self.assertEqual(f.project_calls, [])
                    self.assertEqual(f.score_calls, [])
                    self.assertIs(f.sample_calls[0][0][1], f.parent_result)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + REST, verbosity=2)
