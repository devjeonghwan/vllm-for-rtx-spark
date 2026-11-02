"""Stdlib-only constructor admission regression; no Torch/native/GPU imports."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src/recipes/qwen3_8/helpers/mbx_draft_head2.py"


class UnreadHead:
    @property
    def weight(self):
        raise AssertionError("Invalid geometry must fail before reading the head or preparing a basis")


def fixture():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
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
    exec(compile(code, str(SOURCE), "exec"), namespace)
    return namespace["__init__"], Tensor


class DraftHead2AdmissionTest(unittest.TestCase):
    def test_nonpositive_vocab_or_k_rejected_before_any_head_access(self):
        constructor, _ = fixture()
        for vocab, k in ((0, 1), (-1, 1), (248320, 0), (248320, -1), (0, 0)):
            with self.subTest(vocab=vocab, k=k):
                value = SimpleNamespace(set_basis=lambda _: self.fail("Invalid config prepared basis"))
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    constructor(value, UnreadHead(), vocab, "unused", k, None)

    def test_current_valid_and_above_vocab_clamp_keep_original_dense_path(self):
        constructor, Tensor = fixture()
        for vocab, k, expected in ((248320, 2048, 2048), (17, 100, 17), (1, 1, 1)):
            with self.subTest(vocab=vocab, k=k):
                calls, weight = [], Tensor()
                value = SimpleNamespace(set_basis=lambda path: calls.append(path))
                constructor(value, SimpleNamespace(weight=weight), vocab, "basis-fixture", k, None)
                self.assertIs(value.W, weight)
                self.assertEqual((value.V, value.H, value.k, value.fmt, value.fp4),
                                 (vocab, 2560, expected, "dense", False))
                self.assertEqual(calls, ["basis-fixture"])


if __name__ == "__main__":
    unittest.main()
