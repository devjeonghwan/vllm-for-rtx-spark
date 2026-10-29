"""Fused BF16 HC up projection and gate/mix, with the original Linear fallback."""

import sys
import platform
import torch
from ._ops import hc_upmix

MIN_ROWS = 1024


def _eligible(xn, lora, linear, hc_count, replicated_type, unquantized_type, low_latency_type):
    if (xn.ndim != 2 or not MIN_ROWS <= xn.shape[0] <= 8192 or hc_count != 4
            or type(linear) is not replicated_type or "forward" in vars(linear)
            or getattr(linear, "bias", None) is not None or getattr(linear, "return_bias", None) is not False):
        return False
    method = getattr(linear, "quant_method", None)
    if type(method) not in (unquantized_type, low_latency_type):
        return False
    if getattr(linear, "_forward_hooks", {}) or getattr(linear, "_forward_pre_hooks", {}):
        return False
    if any(getattr(linear, name, None) is not None
           for name in ("weight_scale", "weight_scale_inv", "qweight", "weight_packed", "lora_a", "lora_b")):
        return False
    weight = getattr(linear, "weight", None)
    if (weight is None or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (10240, 320)
            or xn.dtype != torch.bfloat16 or lora.dtype != torch.bfloat16 or lora.ndim != 2
            or tuple(xn.shape) != (lora.shape[0], 10240) or lora.shape[1] != 320):
        return False
    if not (xn.is_cuda and lora.is_cuda and weight.is_cuda and xn.device == lora.device == weight.device):
        return False
    if not (xn.is_contiguous() and weight.is_contiguous() and lora.stride(1) == 1
            and lora.stride(0) in (320, 336)):
        return False
    if any(t.is_neg() or t.is_conj() for t in (xn, lora, weight)):
        return False
    return not (torch.is_grad_enabled() and any(t.requires_grad for t in (xn, lora, weight)))


def hc_upmix_or_original(xn, lora, linear, hc_count, original_mix):
    # Keep all current decoder Graph sizes on their existing pair before imports/admission.
    if xn.ndim != 2 or xn.shape[0] < MIN_ROWS:
        return original_mix(xn, linear(lora), hc_count)
    if sys.platform != "win32" or platform.machine().lower() not in ("arm64", "aarch64"):
        return original_mix(xn, linear(lora), hc_count)
    from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
    from vllm.models.qwen4_exp.nvidia.low_latency_gemm import Qwen4ExpLowLatencyLinearMethod

    if not _eligible(xn, lora, linear, hc_count, ReplicatedLinear, UnquantizedLinearMethod,
                     Qwen4ExpLowLatencyLinearMethod):
        return original_mix(xn, linear(lora), hc_count)
    method = linear.quant_method
    low_latency = type(method) is not UnquantizedLinearMethod
    return hc_upmix(xn, lora, linear.weight, low_latency)
