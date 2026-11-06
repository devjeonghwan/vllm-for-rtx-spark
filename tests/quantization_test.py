"""Stdlib-only constructor admission regression; no Torch/native/GPU imports."""
from __future__ import annotations

import ast
import __future__
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


HEAD_SOURCE = Path(__file__).resolve().parents[1] / "src/recipes/qwen3_8_flash_next/helpers/mbx_draft_head2.py"


class UnreadHead:
    @property
    def weight(self):
        raise AssertionError("Invalid geometry must fail before reading the head or preparing a basis")


def head_fixture():
    tree = ast.parse(HEAD_SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftHead2")
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    bf16, fp16 = object(), object()
    class Tensor:
        shape = (248320, 2560)
        dtype = bf16
        device = "inert-device"
        def dim(self):
            return 2
    namespace = {"torch": SimpleNamespace(Tensor=Tensor, bfloat16=bf16, float16=fp16),
                 "os": SimpleNamespace(environ={})}
    code = ast.fix_missing_locations(ast.Module(body=[constructor], type_ignores=[]))
    exec(compile(code, str(HEAD_SOURCE), "exec"), namespace)
    return namespace["__init__"], Tensor


class DraftHead2AdmissionTest(unittest.TestCase):
    def test_nonpositive_vocab_or_k_rejected_before_any_head_access(self):
        constructor, _ = head_fixture()
        for vocab, k in ((0, 1), (-1, 1), (248320, 0), (248320, -1), (0, 0)):
            with self.subTest(vocab=vocab, k=k):
                value = SimpleNamespace(set_basis=lambda _: self.fail("Invalid config prepared basis"))
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    constructor(value, UnreadHead(), vocab, "unused", k, None)

    def test_current_valid_and_above_vocab_clamp_keep_original_dense_path(self):
        constructor, Tensor = head_fixture()
        for vocab, k, expected in ((248320, 2048, 2048), (17, 100, 17), (1, 1, 1)):
            with self.subTest(vocab=vocab, k=k):
                calls, weight = [], Tensor()
                value = SimpleNamespace(set_basis=lambda path: calls.append(path))
                constructor(value, SimpleNamespace(weight=weight), vocab, "basis-fixture", k, None)
                self.assertIs(value.W, weight)
                self.assertEqual((value.V, value.H, value.k, value.fmt, value.fp4),
                                 (vocab, 2560, expected, "dense", False))
                self.assertEqual(calls, ["basis-fixture"])


QUANT_SOURCE = Path(__file__).resolve().parents[1] / "src/recipes/qwen3_8_flash_next/helpers/mbx_opt_quant.py"

MARLIN_MODULE = "vllm.model_executor.layers.quantization.utils.marlin_utils_fp8"

class MustNotReadBias:
    def __bool__(self):
        raise AssertionError("Bias admission must not read tensor truth/value")

    def __eq__(self, other):
        raise AssertionError("Bias admission must use identity, not tensor equality")

    def item(self):
        raise AssertionError("Bias admission must not synchronize tensor values")

class ReachedWeightConversion(Exception):
    pass

class UnreadWeight:
    @property
    def data(self):
        raise ReachedWeightConversion

def convert_fixture(modules):
    """Compile the real function; all surrounding dependencies are inert fakes."""
    module = ast.parse(QUANT_SOURCE.read_text(encoding="utf-8"))
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "convert")
    calls = []
    fake_marlin = ModuleType(MARLIN_MODULE)

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("Biased modules must not reach Marlin preparation")

    fake_marlin.prepare_fp8_layer_for_marlin = unexpected_prepare
    namespace = {
        "_EXCLUDE": (), "_EXCLUDE_SUFFIX": (),
        "_selected": lambda *args: iter(modules),
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))),
        "log": SimpleNamespace(debug=lambda *args: calls.append(("debug", args)), info=lambda *args: None),
    }
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(tree, str(QUANT_SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["convert"], fake_marlin, calls

class Fp8BiasAdmissionTest(unittest.TestCase):
    def test_biased_modules_keep_original_owners_for_both_bias_contracts(self):
        for separately_returned in (False, True):
            with self.subTest(skip_bias_add=separately_returned):
                weight, bias, method = UnreadWeight(), MustNotReadBias(), object()
                mod = SimpleNamespace(weight=weight, bias=bias, quant_method=method,
                                      skip_bias_add=separately_returned)
                convert, marlin, calls = convert_fixture([("biased", mod)])
                with patch.dict(sys.modules, {MARLIN_MODULE: marlin}):
                    self.assertEqual(convert(object()), 0)
                self.assertIs(mod.weight, weight)
                self.assertIs(mod.bias, bias)
                self.assertIs(mod.quant_method, method)
                self.assertEqual(mod.skip_bias_add, separately_returned)
                self.assertFalse(hasattr(mod, "weight_scale"))
                self.assertTrue(any(isinstance(call, tuple) and call[0] == "debug" for call in calls))

    def test_bias_free_module_reaches_existing_weight_conversion(self):
        mod = SimpleNamespace(weight=UnreadWeight(), bias=None, quant_method=object())
        convert, marlin, _ = convert_fixture([("bias_free", mod)])
        with patch.dict(sys.modules, {MARLIN_MODULE: marlin}):
            with self.assertRaises(ReachedWeightConversion):
                convert(object())

    def test_skipping_bias_does_not_skip_later_bias_free_modules(self):
        biased = SimpleNamespace(weight=UnreadWeight(), bias=MustNotReadBias())
        bias_free = SimpleNamespace(weight=UnreadWeight(), bias=None)
        convert, marlin, calls = convert_fixture([("biased", biased), ("bias_free", bias_free)])
        with patch.dict(sys.modules, {MARLIN_MODULE: marlin}):
            with self.assertRaises(ReachedWeightConversion):
                convert(object())
        self.assertEqual(len([call for call in calls if isinstance(call, tuple) and call[0] == "debug"]), 1)


if __name__ == "__main__":
    unittest.main()
