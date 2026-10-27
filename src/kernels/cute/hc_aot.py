"""Thin Windows ARM64 binding of original generated HC C headers and host COFF.

Original HcDownSiluGemm owns dispatch, output allocation, views and warmup.
No tensor copies, new scratch, arithmetic or Graph implementation is added.
"""
import ctypes as C
import json
import os
from pathlib import Path
import platform
from . import assets

_metadata_cache = None
_kernels = {}


class Tensor0(C.Structure):
    _fields_ = [("data", C.c_void_p)]


class Tensor1(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_shapes", C.c_int32 * 1)]


class Tensor1Stride(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_shapes", C.c_int32 * 1), ("dynamic_strides", C.c_int64 * 1)]


class Tensor2Stride(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_shapes", C.c_int32 * 2), ("dynamic_strides", C.c_int64 * 1)]


def available():
    # CPU-only availability; actual native DLL/context loads happen in warmup.
    return (os.name == "nt" and platform.machine().lower() in ("arm64", "aarch64")
            and bool(os.environ.get("MBX_HC_AOT_MANIFEST")))


def _metadata():
    global _metadata_cache
    if _metadata_cache is None:
        if not available():
            raise RuntimeError("Original HC AOT is not selected for native Windows ARM64")
        path = Path(os.environ["MBX_HC_AOT_MANIFEST"]).resolve()
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if (data["rank"], data["HC"], data["K"], data["logical_N"]) != (320, 4, 10240, 324):
            raise RuntimeError("HC AOT manifest differs from original model geometry")
        assets.verify()
        data["library"] = str(assets.resolve(data["library"]))
        _metadata_cache = data
    return _metadata_cache


class OriginalHcKernel:
    def __init__(self, variant, torch):
        from original_cute_coff_runtime import load_module
        data = _metadata()
        self.variant = variant
        self.torch = torch
        self.module = load_module(data["library"],
            init_symbol=variant["init_symbol"], load_symbol=variant["load_symbol"],
            stream=torch.cuda.current_stream(),
            invoke_symbols={variant["invoke_symbol"]: 5 if variant["backend"] == "fma" else 4})

    def __call__(self, a, b, c, n_dim=None):
        if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
            raise RuntimeError("Original HC host ABI requires two-dimensional tensor views")
        m, k = a.shape
        n = b.shape[0]
        if (k, n) != (10240, 324) or b.shape[1] != k or c.shape != (m, n):
            raise RuntimeError("Original HC AOT tensor geometry differs")
        if any(t.dtype != self.torch.bfloat16 or t.device != a.device or t.stride(1) != 1 for t in (a, b, c)):
            raise RuntimeError("Original HC AOT requires same-device BF16 row-major views")
        stream = C.c_void_p(self.torch.cuda.current_stream(a.device).cuda_stream)
        if self.variant["backend"] == "fma":
            if m != self.variant["M"] or a.stride(0) != k or b.stride(0) != k or n_dim != n:
                raise RuntimeError("Original HC FMA static dimensions/strides changed")
            args = (Tensor0(a.data_ptr()), Tensor1(b.data_ptr(), (C.c_int32 * 1)(n)),
                    Tensor1Stride(c.data_ptr(), (C.c_int32 * 1)(n), (C.c_int64 * 1)(c.stride(0))),
                    C.c_int32(n), stream)
        else:
            low, high = self.variant["M_range"]
            if not low <= m <= high or n_dim is not None:
                raise RuntimeError("Original HC MMA dispatch range changed")
            args = tuple(Tensor2Stride(t.data_ptr(), (C.c_int32 * 2)(*t.shape), (C.c_int64 * 1)(t.stride(0))) for t in (a, b, c)) + (stream,)
        # The original generated host code owns TMA/PDL/grid/block/shared/args.
        # CUDA copies the CPU argument values before this synchronous API returns.
        self.module.invoke(self.variant["invoke_symbol"], [C.addressof(arg) for arg in args])


def get_kernel(compile_key, rank, hc, k, prefetch_pdl_weights=False):
    if (rank, hc, k) != (320, 4, 10240):
        raise RuntimeError("No original HC AOT artifact for this model geometry")
    key = tuple(compile_key)
    if key[0] == "fma" and bool(prefetch_pdl_weights) != (key[1] == 1):
        raise RuntimeError("Original HC PDL weight-prefetch policy changed")
    if key not in _kernels:
        import torch
        matches = [item for item in _metadata()["variants"] if tuple(item["original_dispatch_key"]) == key]
        if len(matches) != 1:
            raise RuntimeError("Missing unique original HC compiler dispatch artifact: " + repr(key))
        _kernels[key] = OriginalHcKernel(matches[0], torch)
    return _kernels[key]
