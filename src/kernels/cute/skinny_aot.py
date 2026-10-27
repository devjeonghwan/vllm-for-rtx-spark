"""Thin native host binding for ORIGINAL Qwen SM121 SkinnyGemmConfig kernels.

Original ShapeDynamicSkinnyGemm retains selection, validation, allocation,
warmup and replay. This module only marshals its generated C header fields.
"""

import ctypes as C
import json
import os
from pathlib import Path
import platform
from . import assets

_manifest = None
_kernels = {}
FAILED_TENSOR_OWNERS = []


class TensorStride(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_strides", C.c_int64 * 1)]


class TensorShapeStride(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_shapes", C.c_int32 * 1),
                ("dynamic_strides", C.c_int64 * 1)]


class TensorTwoShapeStride(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_shapes", C.c_int32 * 2),
                ("dynamic_strides", C.c_int64 * 1)]


def available():
    return (os.name == "nt" and platform.machine().lower() in ("arm64", "aarch64")
            and bool(os.environ.get("MBX_SKINNY_AOT_MANIFEST")))


def _metadata():
    global _manifest
    if _manifest is None:
        if not available():
            raise RuntimeError("Original skinny AOT is not selected for native Windows ARM64")
        path = Path(os.environ["MBX_SKINNY_AOT_MANIFEST"]).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "source_ready_original_SM121_skinny_native_manifest":
            raise RuntimeError("Original skinny AOT manifest is not ready")
        assets.verify()
        manifest["library"] = str(assets.resolve(manifest["library"]))
        _manifest = manifest
    return _manifest


def _key(config):
    fields = ("num_rows", "block_size", "outputs_per_block", "k_unroll", "vector_width", "static_k")
    return tuple(getattr(config, field) for field in fields)


class OriginalSkinnyKernel:
    def __init__(self, variant, torch):
        from original_cute_coff_runtime import load_module
        self.variant, self.torch = variant, torch
        metadata = _metadata()
        if torch.cuda.get_device_capability() != (12, 1) or variant["use_pdl"] is not True:
            raise RuntimeError("Keep the original SM121 PDL policy of this archive")
        self.module = load_module(metadata["library"],
                                  init_symbol=variant["init_symbol"], load_symbol=variant["load_symbol"],
                                  stream=torch.cuda.current_stream(), invoke_symbols={variant["invoke_symbol"]: 5})

    def __call__(self, a, b, residual, output, stream):
        config = self.variant["config"]
        m, k = a.shape
        n = b.shape[0]
        if (m != config["num_rows"] or b.shape[1] != k
                or (config["static_k"] is not None and k != config["static_k"])):
            raise ValueError("Original static SkinnyGemmConfig shape changed")
        if output.shape != (m, n) or residual.shape != (m, n):
            raise ValueError("Original skinny output/residual view shape changed")
        if any(t.dtype != self.torch.bfloat16 or t.device != a.device or t.stride(1) != 1
               for t in (a, b, residual, output)):
            raise ValueError("Original skinny tensors must remain same-device BF16 row-major views")
        # Original Quack uses sym_int32 for dynamic N/K and sym_int64 for
        # row strides, including HC-up's noncontiguous stride336.
        if config["static_k"] is None:
            packed_a = TensorShapeStride(a.data_ptr(), (C.c_int32 * 1)(k), (C.c_int64 * 1)(a.stride(0)))
            packed_b = TensorTwoShapeStride(b.data_ptr(), (C.c_int32 * 2)(n, k), (C.c_int64 * 1)(b.stride(0)))
        else:
            packed_a = TensorStride(a.data_ptr(), (C.c_int64 * 1)(a.stride(0)))
            packed_b = TensorShapeStride(b.data_ptr(), (C.c_int32 * 1)(n), (C.c_int64 * 1)(b.stride(0)))
        arguments = (
            packed_a, packed_b,
            *(TensorShapeStride(t.data_ptr(), (C.c_int32 * 1)(n), (C.c_int64 * 1)(t.stride(0)))
              for t in (residual, output)),
            C.c_void_p(int(getattr(stream, "cuda_stream", stream))),
        )
        try:
            self.module.invoke(self.variant["invoke_symbol"], [C.addressof(arg) for arg in arguments])
        except BaseException:
            FAILED_TENSOR_OWNERS.append((self, a, b, residual, output))
            raise


def get_kernel(dtype, config, has_residual=False, *, use_pdl):
    import torch
    if dtype != torch.bfloat16 or has_residual or use_pdl is not True:
        raise ValueError("Original model archive covers BF16 and its no-residual Qwen dispatch only")
    config_key = _key(config)
    key = torch.cuda.current_device(), config_key
    if key not in _kernels:
        matches = [item for item in _metadata()["variants"] if tuple(item["config_key"]) == config_key]
        if len(matches) != 1:
            raise RuntimeError("Original skinny compile configuration is absent: " + repr(key))
        _kernels[key] = OriginalSkinnyKernel(matches[0], torch)
    return _kernels[key]
