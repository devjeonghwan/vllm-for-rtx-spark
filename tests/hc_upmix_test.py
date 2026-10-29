"""CPU/meta HC admission, source ABI and spawned-process registration contracts."""

import importlib
import json
import multiprocessing
from pathlib import Path
import struct
import sys
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_payload():
    if "_rtx_spark" not in sys.modules:
        package = ModuleType("_rtx_spark")
        package.__path__ = [str(ROOT / "src")]
        sys.modules[package.__name__] = package
    return importlib.import_module("_rtx_spark.kernels.hc_upmix")


def cpu_receipt():
    import torch

    package = load_payload()
    native = importlib.import_module(package.__name__ + "._native")
    xn = torch.empty((1025, 10240), dtype=torch.bfloat16, device="meta")
    lora = torch.empty_strided((1025, 320), (336, 1), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((10240, 320), dtype=torch.bfloat16, device="meta")
    output = package.hc_upmix(xn, lora, weight, True)
    return {"cuda_initialized": torch.cuda.is_initialized(), "native": native.status(),
            "artifact": str(native.asset_path()), "fake_shape": list(output.shape),
            "fake_dtype": str(output.dtype), "operator": str(torch.ops.rtx_spark.hc_upmix.default._schema)}


def worker_receipt(connection):
    try:
        connection.send({"ok": True, "receipt": cpu_receipt()})
    except BaseException as error:
        connection.send({"ok": False, "error": repr(error)})
    finally:
        connection.close()


class TensorMetadata:
    def __init__(self, shape, dtype, *, stride=None, negative=False):
        self.shape, self.dtype, self.ndim = shape, dtype, len(shape)
        self.device, self.is_cuda, self.requires_grad = "cuda:0", True, False
        self._stride, self._negative = stride or (shape[1], 1), negative

    def stride(self, dimension): return self._stride[dimension]
    def is_contiguous(self): return self._stride == (self.shape[1], 1)
    def is_neg(self): return self._negative
    def is_conj(self): return False


class Unquantized:
    pass


class Skinny(Unquantized):
    pass


class UnknownQuant(Unquantized):
    pass


class Replicated:
    def __init__(self, weight):
        self.weight, self.quant_method = weight, Unquantized()
        self.bias, self.return_bias = None, False


class HcUpmix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch, cls.package = torch, load_payload()

    def tensors(self, rows=1024, stride=336):
        dtype = self.torch.bfloat16
        return (TensorMetadata((rows, 10240), dtype), TensorMetadata((rows, 320), dtype, stride=(stride, 1)),
                Replicated(TensorMetadata((10240, 320), dtype)))

    def eligible(self, xn, lora, linear):
        return self.package._eligible(xn, lora, linear, 4, Replicated, Unquantized, Skinny)

    def test_exact_known_linear_methods_and_supported_metadata(self):
        for stride in (320, 336):
            for rows in (1024, 1025, 8192):
                args = self.tensors(rows, stride)
                self.assertTrue(self.eligible(*args))
                args[2].quant_method = Skinny()
                self.assertTrue(self.eligible(*args))

    def test_unknown_quant_bias_wrapper_and_negative_view_reject(self):
        xn, lora, linear = self.tensors()
        linear.quant_method = UnknownQuant()
        self.assertFalse(self.eligible(xn, lora, linear))
        linear.quant_method = Unquantized()
        linear.bias = object()
        self.assertFalse(self.eligible(xn, lora, linear))
        linear.bias = None
        linear.forward = lambda value: value
        self.assertFalse(self.eligible(xn, lora, linear))
        del linear.forward
        xn._negative = True
        self.assertFalse(self.eligible(xn, lora, linear))

    def test_small_rows_do_not_enter_native_or_type_imports(self):
        class Original:
            calls = 0

            def __call__(self, value):
                self.calls += 1
                return "original-gate"

        linear = Original()
        xn = TensorMetadata((513, 10240), self.torch.bfloat16)
        result = self.package.hc_upmix_or_original(xn, None, linear, 4,
                                                  lambda x, gate, count: (gate, count))
        self.assertEqual(result, ("original-gate", 4))
        self.assertEqual(linear.calls, 1)
        native = importlib.import_module(self.package.__name__ + "._native")
        self.assertFalse(native.status()["loaded"])
        self.assertEqual(native.status()["checked_devices"], 0)

    def test_meta_registration_and_direct_operator_rejects_wrong_dtype(self):
        receipt = cpu_receipt()
        self.assertEqual(receipt["fake_shape"], [1025, 2560])
        self.assertFalse(receipt["cuda_initialized"])
        self.assertFalse(receipt["native"]["loaded"])
        self.assertEqual(receipt["native"]["checked_devices"], 0)
        torch = self.torch
        with self.assertRaisesRegex(ValueError, "tensor contract"):
            self.package.hc_upmix(torch.empty((1024, 10240), device="meta"),
                torch.empty((1024, 320), dtype=torch.bfloat16, device="meta"),
                torch.empty((10240, 320), dtype=torch.bfloat16, device="meta"), False)

    def test_native_asset_and_source_contract(self):
        metadata = json.loads((ROOT / "assets/native/manifest.json").read_text(encoding="utf-8"))
        record = metadata["files"]["hc_upmix"]
        self.assertEqual(record["abi"], 2)
        raw = (ROOT / "assets/native" / record["file"]).read_bytes()
        pe = struct.unpack_from("<I", raw, 60)[0]
        self.assertEqual(raw[pe:pe + 6], b"PE\0\0\x64\xaa")
        source = (ROOT / "src/kernels/hc_upmix/native/hc_upmix.cu").read_text(encoding="utf-8")
        self.assertIn("rtx_hc_upmix_abi() { return 2; }", source)
        self.assertIn("lo16 + t * lora_stride", source)
        self.assertIn("__float2bfloat16_rn(acc[mt]", source)
        self.assertNotIn("cudaMalloc", source)
        self.assertNotIn("cudaMemcpy", source)
        self.assertTrue((ROOT / "src/kernels/hc_upmix/LICENSE.strata").is_file())

    def test_spawned_worker_registers_same_cpu_schema_without_CUDA_or_DLL(self):
        expected = cpu_receipt()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        worker = context.Process(target=worker_receipt, args=(child,))
        worker.start()
        child.close()
        try:
            self.assertTrue(parent.poll(30), "CPU registration worker timeout")
            result = parent.recv()
            worker.join(30)
            self.assertEqual(worker.exitcode, 0)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["receipt"], expected)
        finally:
            parent.close()
            if worker.is_alive():
                worker.terminate()
                worker.join(10)


if __name__ == "__main__":
    unittest.main()
