"""CPU admission/selection/source contracts; no Torch, CUDA or model import."""

import ast
from dataclasses import replace
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()
from _rtx_spark.kernels.fp8_fused import enabled  # noqa: E402
from _rtx_spark.kernels.fp8_fused.admission import TensorSpec, admit  # noqa: E402
from _rtx_spark.optimizations import resolve_options  # noqa: E402
from _rtx_spark.optimizations.fp8 import FUSED_FP8  # noqa: E402


class FusedFp8Tests(unittest.TestCase):
    def setUp(self):
        self.a = TensorSpec((17, 3840), (3840, 1), "torch.float8_e4m3fn", "cuda", 0, 256)
        self.b = TensorSpec((3840, 7680), (1, 3840), self.a.dtype, "cuda", 0, 512)
        self.sa = TensorSpec((17, 1), (1, 1), "torch.float32", "cuda", 0, 1024)
        self.sb = TensorSpec((7680, 1), (1, 1), "torch.float32", "cuda", 0, 2048)

    def admission(self, a=None, b=None, sa=None, sb=None, dtype="torch.bfloat16", shape=None, bias=False):
        return admit(a or self.a, b or self.b, sa or self.sa, sb or self.sb,
                     dtype, [1, 7680] if shape is None else shape, bias_present=bias)

    def test_option_default_off_and_engine_selection_has_no_environment_state(self):
        self.assertEqual(resolve_options({}, (FUSED_FP8,)), {"fused_fp8": False})
        for bad in (1, "true", None):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                resolve_options({"fused_fp8": bad}, (FUSED_FP8,))
        for value in (True, False, 1, "true"):
            config = SimpleNamespace(additional_config={"rtx_spark": {"fused_fp8": value}})
            self.assertEqual(enabled(config), value is True)
        self.assertFalse(enabled(SimpleNamespace(additional_config={})))

    def test_original_token_channel_layout_padding_and_reshape_admitted(self):
        self.assertTrue(self.admission()[0])
        self.assertTrue(self.admission(shape=[1, 1, 7680])[0])
        self.assertTrue(self.admission(b=replace(self.b, stride=(7680, 1)))[0])
        self.assertTrue(self.admission(sb=replace(self.sb, shape=(7680,), stride=(1,)))[0])

    def test_actual_m_n_k_tails_are_masked_domain_not_divisibility_rejection(self):
        a = replace(self.a, shape=(3, 257), stride=(257, 1))
        b = replace(self.b, shape=(257, 129), stride=(1, 257))
        sa = replace(self.sa, shape=(3, 1))
        sb = replace(self.sb, shape=(129, 1))
        self.assertTrue(self.admission(a, b, sa, sb, shape=[3, 129])[0])

    def test_bias_always_retains_original_rounding_path(self):
        self.assertFalse(self.admission(bias=True)[0])

    def test_scale_cardinality_layout_dtype_and_device_rejected(self):
        changes = (replace(self.sa, shape=(1,)), replace(self.sa, shape=(17,)),
                   replace(self.sa, stride=(2, 1)), replace(self.sa, dtype="torch.float16"),
                   replace(self.sa, device_index=1), replace(self.sa, data_ptr=1025))
        for value in changes:
            with self.subTest(value=value):
                self.assertFalse(self.admission(sa=value)[0])
        for offset in (4, 8, 12):
            with self.subTest(offset=offset):
                self.assertFalse(self.admission(sa=replace(self.sa, data_ptr=1024 + offset))[0])
                self.assertFalse(self.admission(sb=replace(self.sb, data_ptr=2048 + offset))[0])
        self.assertFalse(self.admission(sb=replace(self.sb, shape=(1, 7680)))[0])

    def test_operand_dtype_stride_view_gradient_and_offset_rejected(self):
        changes = (replace(self.a, dtype="torch.float8_e5m2"), replace(self.a, negative=True),
                   replace(self.a, conjugate=True), replace(self.a, requires_grad=True),
                   replace(self.a, stride=(7680, 2)), replace(self.a, stride=(-3840, 1)),
                   replace(self.a, stride=(2**31, 1)), replace(self.a, device="cpu"),
                   replace(self.a, shape=(0, 3840)), replace(self.a, data_ptr=257))
        for value in changes:
            with self.subTest(value=value):
                self.assertFalse(self.admission(a=value)[0])
        self.assertFalse(self.admission(shape=[18, 7680])[0])
        self.assertFalse(self.admission(dtype="torch.float64")[0])

    def test_device_and_tile_math_ast_preserved_from_attributed_upstream(self):
        directory = ROOT / "src/kernels/fp8_fused"
        trees = [ast.parse((directory / name).read_text(encoding="utf-8"))
                 for name in ("upstream.py", "device.py")]
        for name in ("_fp8_scaled_mm", "_tiles"):
            funcs = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
                     for tree in trees]
            self.assertEqual(ast.dump(funcs[0], include_attributes=False), ast.dump(funcs[1], include_attributes=False))
        self.assertIn("b2de7164a65d1a7cff4ae328236eaf13cc568c13", (directory / "NOTICE").read_text())
        self.assertIn("Apache License", (directory / "LICENSE").read_text())

    def test_compiled_entry_only_dispatches_opaque_operator_without_tensor_guards(self):
        tree = ast.parse((ROOT / "src/kernels/fp8_fused/vllm.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "apply_scaled_mm")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "provider-method", "exec"), namespace)
        marker = object()
        received = []
        instance = SimpleNamespace(_rtx_spark_scaled_mm=lambda *args: received.append(args) or marker)
        tensors = [object() for _ in range(4)]
        self.assertIs(namespace['apply_scaled_mm'](instance, A=tensors[0], B=tensors[1], As=tensors[2],
            Bs=tensors[3], bias=None, out_dtype="BF16", output_shape=[1, 7680]), marker)
        self.assertEqual(received, [(*tensors, None, "BF16", [1, 7680])])

    def test_formal_kernel_selection_requires_engine_option_platform_and_quant_keys(self):
        tree = ast.parse((ROOT / "src/kernels/fp8_fused/vllm.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        config = SimpleNamespace(additional_config={"rtx_spark": {"fused_fp8": True}})
        fake_platform = SimpleNamespace(machine=lambda: "ARM64")
        fake_system = SimpleNamespace(platform="win32")
        namespace = {'ChannelWiseTorchFP8ScaledMMLinearKernel': type('OriginalKernel', (), {}),
                     'torch': SimpleNamespace(bfloat16="BF16", float16="F16", float32="F32"),
                     'get_current_vllm_config': lambda: config, 'enabled': enabled,
                     'sys': fake_system, 'platform': fake_platform,
                     'kFp8DynamicTokenSym': "TOKEN", 'kFp8StaticChannelSym': "CHANNEL"}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'formal-provider', 'exec'), namespace)
        provider = namespace[cls.name]
        self.assertTrue(provider.is_supported(121)[0])
        self.assertFalse(provider.is_supported(120)[0])
        config.additional_config = {}
        self.assertFalse(provider.is_supported(121)[0])
        config.additional_config = {"rtx_spark": {"fused_fp8": True}}
        fake_system.platform = "linux"
        self.assertFalse(provider.is_supported(121)[0])
        fake_system.platform = "win32"
        fake_platform.machine = lambda: "AMD64"
        self.assertFalse(provider.is_supported(121)[0])
        layer = SimpleNamespace(activation_quant_key="TOKEN", weight_quant_key="CHANNEL", out_dtype="BF16")
        self.assertTrue(provider.can_implement(layer)[0])
        layer.weight_quant_key = "BLOCK"
        self.assertFalse(provider.can_implement(layer)[0])
        layer.weight_quant_key = "CHANNEL"
        layer.activation_quant_key = "STATIC_TENSOR"
        self.assertFalse(provider.can_implement(layer)[0])

    def test_capture_miss_and_current_device_mismatch_fall_back_before_launch(self):
        tree = ast.parse((ROOT / "src/kernels/fp8_fused/ops.py").read_text())
        defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in
                ("implementation", "variant_key")]
        marker = object()
        namespace = {'TensorSpec': SimpleNamespace(of=lambda x: x.spec), 'admit': admit,
            '_READY': set(), '_CAPABILITIES': {}, 'fallback': lambda *args: marker}
        fake_cuda = SimpleNamespace(current_device=lambda: 1,
                                    is_current_stream_capturing=lambda: True)
        namespace['torch'] = SimpleNamespace(cuda=fake_cuda)
        exec(compile(ast.Module(body=defs, type_ignores=[]), "opaque-runtime", "exec"), namespace)
        class Tensor:
            def __init__(self, spec):
                self.spec = spec
                self.shape = spec.shape
                self.device = SimpleNamespace(index=spec.device_index)
            def stride(self):
                return self.spec.stride
        values = [Tensor(spec) for spec in (self.a, self.b, self.sa, self.sb)]
        self.assertIs(namespace['implementation'](*values, None, 'torch.bfloat16', [1, 7680]), marker)
        fake_cuda.current_device = lambda: 0
        self.assertIs(namespace['implementation'](*values, None, 'torch.bfloat16', [1, 7680]), marker)
        self.assertEqual(namespace['_READY'], set())
        self.assertEqual(namespace['_CAPABILITIES'], {})


if __name__ == "__main__":
    unittest.main()
