"""CPU admission, fallback and pinned-kernel regeneration contracts."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()
from _rtx_spark.runtime import sampling  # noqa: E402
from _rtx_spark.kernels import sampling as kernels  # noqa: E402

GENERATOR = runpy.run_path(str(ROOT / "scripts/generate_sampling_kernels.py"))
KERNELS = ROOT / "src/kernels/sampling"


def array(values):
    return SimpleNamespace(np=np.array(values), gpu=object())


def fixture():
    logits = SimpleNamespace(
        ndim=2,
        shape=(8, 2051),
        dtype="torch.bfloat16",
        stride=lambda axis: 1,
        device=SimpleNamespace(type="cuda"),
    )
    states = SimpleNamespace(
        num_logprobs=np.array([-1] * 4),
        top_k=array([20] * 4),
        top_p=array([0.95] * 4),
        temperature=array([0.6, 1, 1.3, 0.9]),
        min_p=array([0.0] * 4),
        seeds=array([10, 20, 30, 40]),
    )

    def processor(name, attribute, value):
        cls = type(name, (), {"__module__": "vllm.v1.worker.gpu.sample.test"})
        item = cls()
        setattr(item, attribute, value)
        return item

    processors = [
        processor("LogitBiasState", "use_logit_bias", np.array([False] * 4)),
        processor("PenaltiesState", "use_penalty", np.array([False] * 4)),
        processor("BadWordsState", "num_bad_words", array([0] * 4)),
    ]
    sampler = SimpleNamespace(
        trace_replay_state=None,
        return_sampling_mask=False,
        logprobs_mode="raw_logprobs",
        sampling_states=states,
        logits_processors=processors,
        use_fp64_gumbel=False,
        thinking_budget_state=SimpleNamespace(enabled=False),
    )
    owner = SimpleNamespace(
        sampler=sampler,
        use_block_verification=True,
        synthetic_conditional_rates=None,
        watermark_key=None,
        num_speculative_steps=7,
    )
    return owner, logits, np.array([2, 0])


def import_worker(connection):
    """Spawn imports must not initialize Torch, Triton or generated kernels."""
    try:
        public = importlib.import_module("_rtx_spark.runtime.sampling")
        package = importlib.import_module("_rtx_spark.kernels.sampling")
        connection.send(
            {
                "ok": True,
                "eligible": callable(public.eligible),
                "loaded": [name for name in ("torch", "triton") if name in sys.modules],
                "lazy": package._TOPK is None and package._REJECTION is None,
            }
        )
    except BaseException as error:
        connection.send({"ok": False, "error": repr(error)})
    finally:
        connection.close()


def source_root(test):
    value = os.environ.get("VLLM_RTX_SPARK_SOURCE_DIR")
    if not value:
        test.skipTest("Set VLLM_RTX_SPARK_SOURCE_DIR to the freshly prepared vLLM source")
    path = Path(value)
    for relative in GENERATOR["INPUTS"].values():
        test.assertTrue((path / relative).is_file(), str(path / relative))
    return path


def verify_function(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RejectionSampler"
    )
    return next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_verify")


class SamplingTests(unittest.TestCase):
    def test_noncontiguous_request_slots_use_selected_state(self):
        owner, logits, idx = fixture()
        owner.sampler.sampling_states.min_p.np[1] = 0.5
        self.assertEqual(sampling.eligible(owner, logits, idx), (True, "eligible"))
        owner.sampler.sampling_states.min_p.np[2] = 0.5
        self.assertEqual(sampling.eligible(owner, logits, idx), (False, "min_p"))

    def test_processed_modes_fall_back_across_chunk_boundaries(self):
        for mode in ("processed_logprobs", "processed_logits"):
            owner, logits, idx = fixture()
            owner.sampler.logprobs_mode = mode
            owner.sampler.sampling_states.num_logprobs[1] = 5
            self.assertTrue(np.all(owner.sampler.sampling_states.num_logprobs[idx] == -1))
            self.assertEqual(sampling.eligible(owner, logits, idx), (False, "processed_logprobs_mode"))

    def test_output_and_verification_consumers_fall_back(self):
        for mode in ("logprobs", "mask", "trace", "watermark", "synthetic", "nonblock"):
            with self.subTest(mode=mode):
                owner, logits, idx = fixture()
                if mode == "logprobs":
                    owner.sampler.sampling_states.num_logprobs[2] = 0
                elif mode == "mask":
                    owner.sampler.return_sampling_mask = True
                elif mode == "trace":
                    owner.sampler.trace_replay_state = object()
                elif mode == "watermark":
                    owner.watermark_key = 123
                elif mode == "synthetic":
                    owner.synthetic_conditional_rates = object()
                else:
                    owner.use_block_verification = False
                self.assertFalse(sampling.eligible(owner, logits, idx)[0])

    def test_active_or_custom_processors_fall_back(self):
        for index, attribute in enumerate(("use_logit_bias", "use_penalty", "num_bad_words")):
            owner, logits, idx = fixture()
            values = getattr(owner.sampler.logits_processors[index], attribute)
            if attribute == "num_bad_words":
                values = values.np
            values[2] = 1
            self.assertFalse(sampling.eligible(owner, logits, idx)[0], attribute)
        owner, logits, idx = fixture()
        owner.sampler.logits_processors.append(object())
        self.assertEqual(sampling.eligible(owner, logits, idx), (False, "custom_processors"))
        owner, logits, idx = fixture()
        type(owner.sampler.logits_processors[0]).__module__ = "external.processor"
        self.assertEqual(sampling.eligible(owner, logits, idx), (False, "custom_processors"))

    def test_sampling_parameter_boundaries(self):
        for field, invalid in (
            ("temperature", [0, -1, np.nan, np.inf]),
            ("top_p", [0, -0.1, 1.1, np.nan, np.inf]),
            ("top_k", [0, -1, 2051, 2052]),
            ("min_p", [0.01]),
        ):
            for value in invalid:
                with self.subTest(field=field, value=value):
                    owner, logits, idx = fixture()
                    getattr(owner.sampler.sampling_states, field).np[2] = value
                    self.assertFalse(sampling.eligible(owner, logits, idx)[0])
        owner, logits, idx = fixture()
        owner.sampler.thinking_budget_state = SimpleNamespace(
            enabled=True, use_thinking_budget=np.array([0, 0, 1, 0])
        )
        self.assertEqual(sampling.eligible(owner, logits, idx), (False, "thinking_budget"))
        owner, logits, idx = fixture()
        owner.sampler.sampling_states.top_p.np[:] = 1
        owner.sampler.sampling_states.temperature.np[:] = 1
        self.assertTrue(sampling.eligible(owner, logits, idx)[0])

    def test_layout_and_dtype_fallback(self):
        for field, value in (
            ("ndim", 1),
            ("shape", (0, 2051)),
            ("dtype", "torch.float64"),
            ("stride", lambda axis: 2),
            ("device", SimpleNamespace(type="cpu")),
        ):
            owner, logits, idx = fixture()
            setattr(logits, field, value)
            self.assertFalse(sampling.eligible(owner, logits, idx)[0], field)
        for dtype in ("torch.float32", "torch.float16", "torch.bfloat16"):
            owner, logits, idx = fixture()
            logits.dtype = dtype
            self.assertTrue(sampling.eligible(owner, logits, idx)[0], dtype)

    def test_ineligible_dispatch_never_calls_kernels_or_mutates_inputs(self):
        owner, logits, idx = fixture()
        owner.sampler.logprobs_mode = "processed_logprobs"
        tail = [object() for _ in range(9)]
        args = (
            owner,
            logits,
            tail[0],
            tail[1],
            tail[2],
            tail[3],
            tail[4],
            idx,
            tail[5],
            tail[6],
            tail[7],
            tail[8],
        )
        with patch.object(kernels, "make_filter", side_effect=AssertionError("fallback called a kernel")):
            self.assertIsNone(sampling.try_verify(*args))
        self.assertEqual(logits.shape, (8, 2051))
        self.assertEqual(owner.sampler.sampling_states.num_logprobs.tolist(), [-1] * 4)

    def test_supported_dispatch_preserves_invalid_draft_and_rng_arguments(self):
        owner, logits, idx = fixture()
        owner.sampler.use_fp64_gumbel = True
        draft_logits, draft, pos, cu, mapping, expanded, local, lengths, invalid = [
            object() for _ in range(9)
        ]
        k, p, bits, sampled, count = [object() for _ in range(5)]
        states = owner.sampler.sampling_states
        states.get_top_k_top_p = lambda *args: (k, p)
        with (
            patch.object(kernels, "make_filter", return_value=bits) as make_filter,
            patch.object(kernels, "masked_rejection", return_value=(sampled, count)) as reject,
        ):
            actual = sampling.try_verify(
                owner, logits, draft_logits, draft, pos, cu, mapping, idx, expanded, local, lengths, invalid
            )
        self.assertEqual(actual, (logits, sampled, count))
        make_filter.assert_called_once_with(logits, expanded, states.temperature.gpu, k, p)
        reject.assert_called_once_with(
            logits,
            draft_logits,
            invalid,
            cu,
            pos,
            mapping,
            expanded,
            local,
            states.temperature.gpu,
            states.seeds.gpu,
            7,
            bits,
            use_fp64=True,
            use_block_verification=True,
        )

    def test_spawn_import_is_cpu_only_and_lazy(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=import_worker, args=(child,))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(30), "CPU import worker timed out")
            result = parent.recv()
            process.join(30)
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(result, {"ok": True, "eligible": True, "loaded": [], "lazy": True})
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(10)

    def test_topk_never_writes_input_and_retains_rounding_and_ties(self):
        source = (KERNELS / "topk.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.store":
                self.assertNotIn("LOGITS_ROW", ast.unparse(node.args[0]))
        for required in (
            "mov.b32",
            "duplicate_count",
            "packed_bits",
            "Park et al.",
            "SPDX-FileCopyrightText: Copyright contributors to the vLLM project",
        ):
            self.assertIn(required, source)

    def test_generated_sources_match_the_explicit_prepared_source(self):
        generated, _ = GENERATOR["generate"](source_root(self))
        for name, source in generated.items():
            self.assertEqual((KERNELS / (name + ".py")).read_text(encoding="utf-8"), source)

    def test_Qrita_search_and_duplicate_algorithm_is_unchanged(self):
        source = source_root(self) / GENERATOR["INPUTS"]["topk"]
        original = ast.parse(source.read_text(encoding="utf-8"))
        expected = next(
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name == "_topk_topp_kernel"
        )
        actual = next(
            node
            for node in ast.parse((KERNELS / "topk.py").read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "_rtx_topk_topp_mask_kernel"
        )
        actual.name = expected.name
        actual.args.args = actual.args.args[:-4]
        outer = next(node for node in actual.body if isinstance(node, ast.For))
        outer.body = outer.body[2:-2]
        original_loop = next(node for node in expected.body if isinstance(node, ast.For))
        outer.body.append(deepcopy(original_loop.body[-1]))

        class Inverse(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                return node.args[0] if ast.unparse(node.func) == "_rtx_scale_logits" else node

        actual = Inverse().visit(actual)
        self.assertEqual(
            ast.dump(expected, include_attributes=False), ast.dump(actual, include_attributes=False)
        )

    def test_inverse_target_loads_reconstruct_every_original_rejection_function(self):
        source = source_root(self) / GENERATOR["INPUTS"]["rejection"]
        original = ast.parse(source.read_text(encoding="utf-8"))
        tree = ast.parse((KERNELS / "rejection.py").read_text(encoding="utf-8"))
        extra = {"_rtx_filter", "_rtx_filter_stride", "_rtx_mapping", "_rtx_temperature"}

        class Inverse(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if ast.unparse(node.func) == "_rtx_target_load":
                    pointer, row, column, stride = node.args[:4]
                    offset = ast.BinOp(
                        left=ast.BinOp(
                            left=pointer, op=ast.Add(), right=ast.BinOp(left=row, op=ast.Mult(), right=stride)
                        ),
                        op=ast.Add(),
                        right=column,
                    )
                    return ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="tl", ctx=ast.Load()), attr="load", ctx=ast.Load()
                        ),
                        args=[offset],
                        keywords=node.keywords,
                    )
                node.keywords = [keyword for keyword in node.keywords if keyword.arg not in extra]
                return node

            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                node.args.args = [arg for arg in node.args.args if arg.arg not in extra]
                pairs = [
                    (arg, default)
                    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if arg.arg not in extra
                ]
                node.args.kwonlyargs = [pair[0] for pair in pairs]
                node.args.kw_defaults = [pair[1] for pair in pairs]
                return node

        tree = Inverse().visit(tree)
        actual = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for expected in original.body:
            if isinstance(expected, ast.FunctionDef):
                self.assertEqual(
                    ast.dump(expected, include_attributes=False),
                    ast.dump(actual[expected.name], include_attributes=False),
                    expected.name,
                )

    def test_source_hook_keeps_the_original_fallback_body_and_arguments(self):
        function = verify_function(source_root(self) / "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py")
        self.assertEqual(
            ast.unparse(function.body[0]), "from vllm._rtx_spark.runtime.sampling import try_verify"
        )
        hook_call = function.body[1].value
        self.assertEqual([arg.id for arg in hook_call.args], [arg.arg for arg in function.args.args])
        self.assertEqual(
            ast.unparse(function.body[2]), "if fused_result is not None:\n    return fused_result"
        )
        owner, logits, idx = fixture()
        owner.sampler.logprobs_mode = "processed_logits"
        tail = [object() for _ in range(9)]
        args = (
            owner,
            logits,
            tail[0],
            tail[1],
            tail[2],
            tail[3],
            tail[4],
            idx,
            tail[5],
            tail[6],
            tail[7],
            tail[8],
        )
        calls, processed, sampled, count = [], object(), object(), object()
        owner.sampler.apply_sampling_params = lambda *values: (calls.append(("params", values)), processed)[1]
        owner._watermarking_kwargs = lambda *values: {"marker": values}
        namespace = {
            "rejection_sample": lambda *values, **kw: (
                calls.append(("reject", values, kw)),
                (sampled, count),
            )[1]
        }
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        import __future__

        with patch.dict(sys.modules, {"vllm._rtx_spark.runtime.sampling": sampling}):
            exec(
                compile(module, "prepared-sampler-hook", "exec", flags=__future__.annotations.compiler_flag),
                namespace,
            )
            result = namespace["_verify"](*args)
        self.assertEqual(result, (processed, sampled, count))
        self.assertEqual(
            calls[0], ("params", (logits, tail[5], tail[4], idx, tail[2], tail[1], tail[6], tail[7]))
        )
        self.assertEqual(
            calls[1][1],
            (
                processed,
                tail[0],
                tail[8],
                tail[3],
                tail[2],
                tail[4],
                tail[5],
                tail[6],
                owner.sampler.sampling_states.temperature.gpu,
                owner.sampler.sampling_states.seeds.gpu,
                7,
                None,
            ),
        )
        self.assertEqual(
            calls[1][2],
            {"use_fp64": False, "use_block_verification": True, "marker": (tail[1], tail[5], tail[6])},
        )


if __name__ == "__main__":
    unittest.main()
