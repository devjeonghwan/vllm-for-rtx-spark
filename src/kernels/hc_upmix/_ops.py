"""Torch schema/fake registration; real execution owns all native/device access."""

import torch
from . import _native


def _check_inputs(xn, lora, weight):
    if (xn.ndim != 2 or lora.ndim != 2 or weight.ndim != 2 or not 1 <= xn.shape[0] <= 8192
            or tuple(xn.shape) != (lora.shape[0], 10240) or lora.shape[1] != 320
            or tuple(weight.shape) != (10240, 320)
            or any(t.dtype != torch.bfloat16 for t in (xn, lora, weight))
            or any(t.is_neg() or t.is_conj() for t in (xn, lora, weight))
            or not (xn.device == lora.device == weight.device)
            or not (xn.is_contiguous() and weight.is_contiguous() and lora.stride(1) == 1
                    and lora.stride(0) in (320, 336))):
        raise ValueError("Unsupported RTX Spark HC tensor contract")


@torch.library.custom_op("rtx_spark::hc_upmix", mutates_args=())
def hc_upmix(xn: torch.Tensor, lora: torch.Tensor, weight: torch.Tensor,
             low_latency: bool) -> torch.Tensor:
    _check_inputs(xn, lora, weight)
    if not all(t.is_cuda for t in (xn, lora, weight)):
        raise ValueError("RTX Spark HC native execution requires CUDA tensors")
    if not _native.available_device(xn) or not _native.aligned(xn, lora, weight):
        gate = (torch.ops.vllm.qwen4_exp_low_latency_gemm(lora, weight) if low_latency
                else torch.nn.functional.linear(lora, weight))
        return torch.ops.vllm.qwen4_exp_hc_gate_mix(xn, gate, 4)
    return _native.invoke(xn, lora, weight)


@hc_upmix.register_fake
def hc_upmix_fake(xn, lora, weight, low_latency):
    _check_inputs(xn, lora, weight)
    return xn.new_empty((xn.shape[0], 2560))
