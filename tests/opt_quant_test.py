"""CPU-only conversion admission regression; imports no Torch or GPU modules."""
from __future__ import annotations

import ast
import __future__
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "src/recipes/qwen3_8/helpers/mbx_opt_quant.py"
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
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
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
    exec(compile(tree, str(SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
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
