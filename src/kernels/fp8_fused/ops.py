"""Opaque guarded launch. Tensor/value guards never enter a compiled forward."""

from math import prod

import torch

from .admission import TensorSpec, admit

# CPU metadata only. No input, output, scale or workspace tensors are retained.
_READY = set()
_CAPABILITIES = {}


def fallback(A, B, As, Bs, bias, out_dtype, output_shape):
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    )
    # The pinned original method does not read self. Call it directly so all
    # unsupported cases retain the same FP32 GEMM/scaling/bias/rounding path.
    return ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm(
        None, A=A, B=B, As=As, Bs=Bs, bias=bias, out_dtype=out_dtype, output_shape=output_shape)


def variant_key(A, B, As, Bs, out_dtype):
    return (A.device.index, tuple(A.shape), tuple(B.shape), tuple(A.stride()), tuple(B.stride()),
            tuple(As.shape), tuple(Bs.shape), tuple(As.stride()), tuple(Bs.stride()), str(out_dtype))


def implementation(A, B, As, Bs, bias, out_dtype, output_shape):
    accepted, _ = admit(*(TensorSpec.of(t) for t in (A, B, As, Bs)), str(out_dtype),
                         output_shape, bias_present=bias is not None)
    if not accepted:
        return fallback(A, B, As, Bs, bias, out_dtype, output_shape)
    if torch.cuda.current_device() != A.device.index:
        return fallback(A, B, As, Bs, bias, out_dtype, output_shape)
    capturing = torch.cuda.is_current_stream_capturing()
    key = variant_key(A, B, As, Bs, out_dtype)
    if capturing and key not in _READY:
        return fallback(A, B, As, Bs, bias, out_dtype, output_shape)
    device = A.device.index
    if device not in _CAPABILITIES:
        if capturing:
            return fallback(A, B, As, Bs, bias, out_dtype, output_shape)
        _CAPABILITIES[device] = torch.cuda.get_device_capability(A.device)
    if _CAPABILITIES[device] != (12, 1):
        return fallback(A, B, As, Bs, bias, out_dtype, output_shape)
    # Imports/JIT are forbidden on a first capture. Readiness is established by
    # a successful eager launch of this exact metadata variant during warmup.
    try:
        from .device import _fp8_scaled_mm, _tiles
    except ImportError:
        return fallback(A, B, As, Bs, bias, out_dtype, output_shape)
    m, k = A.shape
    n = B.shape[1]
    result = torch.empty((m, n), dtype=out_dtype, device=A.device)
    bm, bn, bk, warps, stages = _tiles(m, k)
    grid = ((m + bm - 1) // bm * ((n + bn - 1) // bn),)
    _fp8_scaled_mm[grid](A, B, As, Bs, result, m, n, k,
                         A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                         result.stride(0), result.stride(1), BM=bm, BN=bn, BK=bk,
                         GROUP_M=8, num_warps=warps, num_stages=stages)
    if not capturing:
        _READY.add(key)
    return result.narrow(0, 0, prod(output_shape[:-1])).view(output_shape)


@torch.library.custom_op("rtx_spark_fp8::scaled_mm", mutates_args=())
def scaled_mm(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor,
              bias: torch.Tensor | None, out_dtype: torch.dtype,
              output_shape: list[int]) -> torch.Tensor:
    return implementation(A, B, As, Bs, bias, out_dtype, output_shape)


@scaled_mm.register_fake
def _fake(A, B, As, Bs, bias, out_dtype, output_shape):
    return A.new_empty(output_shape, dtype=out_dtype)
