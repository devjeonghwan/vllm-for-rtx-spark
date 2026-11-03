# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused Triton kernel for DFlash KV materialization.

Combines: KV projection + RMSNorm + RoPE, then pool-managed KV writes.
"""

from typing import Callable, List, Optional

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit
def _fused_norm_rope_kernel_stacked(
    kv_ptr,  # [total_ctx, n_layers, kv_size * 2]
    k_norm_weight_ptr,  # [n_layers, head_dim]
    eps_ptr,  # [n_layers]
    cos_sin_cache_ptr,  # [max_pos, rotary_dim]
    positions_ptr,  # [total_ctx]
    k_out_ptr,  # [n_layers, total_ctx, num_kv_heads, head_dim]
    v_out_ptr,  # [n_layers, total_ctx, num_kv_heads, head_dim]
    kv_stride_ctx,
    kv_stride_layer,
    k_norm_weight_stride_layer,
    cos_sin_stride_pos,
    k_out_stride_layer,
    k_out_stride_ctx,
    k_out_stride_head,
    v_out_stride_layer,
    v_out_stride_ctx,
    v_out_stride_head,
    total_ctx,
    n_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """Fused RMSNorm(K) + RoPE(K) materialization. Grid: (total_ctx, num_kv_heads, n_layers)."""
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)
    layer_id = tl.program_id(2)
    if ctx_id >= total_ctx or layer_id >= n_layers:
        return

    position = tl.load(positions_ptr + ctx_id)
    eps = tl.load(eps_ptr + layer_id).to(tl.float32)
    kv_base = kv_ptr + ctx_id * kv_stride_ctx + layer_id * kv_stride_layer
    k_base = kv_base + head_id * head_dim
    v_base = kv_base + kv_size + head_id * head_dim
    k_write = (
        k_out_ptr
        + layer_id * k_out_stride_layer
        + ctx_id * k_out_stride_ctx
        + head_id * k_out_stride_head
    )
    v_write = (
        v_out_ptr
        + layer_id * v_out_stride_layer
        + ctx_id * v_out_stride_ctx
        + head_id * v_out_stride_head
    )

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    mask_half = offs < half_rotary_dim

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(
        k_norm_weight_ptr + layer_id * k_norm_weight_stride_layer + offs,
        mask=mask_hd,
        other=1.0,
    ).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr
        + layer_id * k_norm_weight_stride_layer
        + half_rotary_dim
        + offs,
        mask=mask_half,
        other=1.0,
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    tl.store(v_write + offs, v_raw, mask=mask_hd)
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)
    tl.store(
        k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half
    )
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)


def _fused_norm_rope_stacked(
    kv: torch.Tensor,  # [total_ctx, n_layers, kv_size*2]
    k_norm_weight: torch.Tensor,  # [n_layers, head_dim]
    eps: torch.Tensor,  # [n_layers]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    positions: torch.Tensor,  # [total_ctx]
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + RoPE materialization for all layers."""
    if kv.ndim != 3:
        raise ValueError(
            "Invalid stacked fused KV projection shape: "
            f"got {tuple(kv.shape)}, expected 3D [total_ctx, n_layers, kv_size*2]."
        )

    total_ctx, n_layers, kv_dim = kv.shape
    if total_ctx == 0:
        empty = torch.empty(
            (n_layers, 0, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
        )
        return empty, empty

    kv_size = num_kv_heads * head_dim
    if kv_dim != kv_size * 2:
        raise ValueError(
            "Invalid fused KV projection shape: "
            f"got {tuple(kv.shape)}, expected trailing dim {kv_size * 2}."
        )
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            "Invalid fused KV rotary/head dim pair: "
            f"rotary_dim={rotary_dim}, head_dim={head_dim}."
        )
    if k_norm_weight.shape != (n_layers, head_dim):
        raise ValueError(
            "Invalid stacked k_norm_weight shape for fused KV materialization: "
            f"got {tuple(k_norm_weight.shape)}, expected {(n_layers, head_dim)}."
        )
    if eps.shape != (n_layers,):
        raise ValueError(
            "Invalid stacked eps shape for fused KV materialization: "
            f"got {tuple(eps.shape)}, expected {(n_layers,)}."
        )

    half_rotary_dim = rotary_dim // 2
    BLOCK_HD = triton.next_power_of_2(head_dim)

    if positions.device != kv.device:
        positions = positions.to(device=kv.device, dtype=torch.int64)
    elif positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    expected_shape = (n_layers, total_ctx, num_kv_heads, head_dim)
    if k_out is None:
        k_out = torch.empty(expected_shape, dtype=kv.dtype, device=kv.device)
    else:
        if k_out.shape != expected_shape:
            raise ValueError(
                "Invalid k_out shape for fused KV materialization: "
                f"got {tuple(k_out.shape)}, expected {expected_shape}."
            )
        if k_out.device != kv.device or k_out.dtype != kv.dtype:
            raise ValueError(
                "Invalid k_out device/dtype for fused KV materialization: "
                f"got device={k_out.device}, dtype={k_out.dtype}, "
                f"expected device={kv.device}, dtype={kv.dtype}."
            )
    if v_out is None:
        v_out = torch.empty_like(k_out)
    else:
        if v_out.shape != expected_shape:
            raise ValueError(
                "Invalid v_out shape for fused KV materialization: "
                f"got {tuple(v_out.shape)}, expected {expected_shape}."
            )
        if v_out.device != kv.device or v_out.dtype != kv.dtype:
            raise ValueError(
                "Invalid v_out device/dtype for fused KV materialization: "
                f"got device={v_out.device}, dtype={v_out.dtype}, "
                f"expected device={kv.device}, dtype={kv.dtype}."
            )

    _fused_norm_rope_kernel_stacked[(total_ctx, num_kv_heads, n_layers)](
        kv,
        k_norm_weight,
        eps,
        cos_sin_cache,
        positions,
        k_out,
        v_out,
        kv.stride(0),
        kv.stride(1),
        k_norm_weight.stride(0),
        cos_sin_cache.stride(0),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        total_ctx,
        n_layers,
        num_kv_heads,
        head_dim,
        kv_size,
        rotary_dim,
        half_rotary_dim,
        BLOCK_HD,
    )
    return k_out, v_out


# vLLM adapter below. The two functions above are verbatim SGLang
# 3c560089e1818a5217a73f2168273485a5350341 (also identical at a4e51e2873).
# The adapter keeps vLLM's target-feature projection, hidden RMSNorm and cache
# writer. Only context-K normalization+RoPE uses that original SGLang kernel.
SOURCE_REVISION = "3c560089e1818a5217a73f2168273485a5350341"
SOURCE_PATH = "python/sglang/kernels/ops/speculative/fused_kv_materialize.py"


def prepare(model, capacity: int) -> None:
    """Own all scratch once after draft weights load, before Graph capture.

    One serial GPUModelRunner owns this state. Graphs retain stable addresses;
    no growth, CPU copies, stream changes or synchronizations occur at forward.
    The source model and these buffers must outlive every captured graph.
    """
    if hasattr(model, "_dflash2_sg_kv"):
        raise RuntimeError("DFlash2 KV workspace was already prepared")
    if tuple(model.config.architectures or ()) != ("DFlash2DraftModel",):
        raise NotImplementedError("Fusion admits the external DFlash2 draft only")
    if capacity <= 0 or not isinstance(capacity, int):
        raise ValueError("Positive fixed workspace capacity required")
    weight = model._fused_kv_weight
    if weight is None or model._fused_kv_bias is not None:
        raise NotImplementedError("Fusion requires the existing unquantized bias-free batched KV weight")
    if weight.dtype != torch.bfloat16 or weight.device.type != "cuda":
        raise NotImplementedError("Current native DFlash2 candidate requires CUDA BF16 draft")
    layers = model._attn_layers
    count, heads, dimension = model._num_attn_layers, model._num_kv_heads, model._head_dim
    if (count, heads, dimension, len(layers)) != (5, 8, 128, 5):
        raise NotImplementedError("Only the trained five-layer/KV8/head128 draft is admitted")
    if not model._rope_is_neox:
        raise NotImplementedError("The original SGLang kernel admits NeoX RoPE only")
    rotary = model.layers[0].self_attn.rotary_emb
    rotary_dim = int(rotary.rotary_dim)
    if rotary_dim != dimension:
        raise NotImplementedError("This candidate requires the original full-head draft RoPE")
    hidden = weight.shape[1]
    expected_out = count * 2 * heads * dimension
    if tuple(weight.shape) != (expected_out, 2560) or hidden != model.config.hidden_size:
        raise ValueError("Existing batched KV weight layout changed")
    if tuple(model._k_norm_weights.shape) != (count, dimension):
        raise ValueError("Grouped K norm weights have the wrong layout")
    # Retain the current vLLM cache precision; no new rotary values/override.
    cosine = model._rope_cos_sin_cache.to(device=weight.device, dtype=weight.dtype)
    model._dflash2_sg_kv = {
        "capacity": capacity, "device": weight.device, "dtype": weight.dtype,
        "layers": count, "heads": heads, "head_dim": dimension, "rotary_dim": rotary_dim,
        "weight": weight, "norm_weight": model._k_norm_weights,
        "cos_sin": cosine,
        "eps": torch.full((count,), model._rms_norm_eps, dtype=torch.float32, device=weight.device),
        "normed": torch.empty((capacity, hidden), dtype=weight.dtype, device=weight.device),
        "projected": torch.empty((capacity, expected_out), dtype=weight.dtype, device=weight.device),
        "key": torch.empty((count, capacity, heads, dimension), dtype=weight.dtype, device=weight.device),
        "value": torch.empty((count, capacity, heads, dimension), dtype=weight.dtype, device=weight.device),
    }


def materialize(model, context_states: torch.Tensor, context_positions: torch.Tensor):
    """Return stable scratch views with the original SG norm+RoPE arithmetic.

    This deliberately does not promise bit equality to vLLM's intermediate BF16
    K-norm rounding. Probe comparisons must use the predeclared numerical bound.
    """
    from vllm import _custom_ops as ops
    state = getattr(model, "_dflash2_sg_kv", None)
    if state is None:
        raise RuntimeError("Prepare fixed KV workspace before first Graph/forward")
    num_ctx = context_states.shape[0]
    if context_states.ndim != 2 or context_states.shape[1] != 2560:
        raise ValueError("DFlash2 context must be [N,2560]")
    if not 0 <= num_ctx <= state["capacity"]:
        raise ValueError("DFlash2 context exceeds the frozen Graph workspace capacity")
    if context_states.dtype != state["dtype"] or context_states.device != state["device"]:
        raise ValueError("Context dtype/device differs from the fixed workspace")
    if context_positions.ndim != 1 or context_positions.shape[0] != num_ctx:
        raise ValueError("Context positions must match every context row")
    if context_positions.dtype != torch.int64 or context_positions.device != state["device"]:
        raise ValueError("Reuse existing device int64 positions; forward conversion is forbidden")
    if context_positions.stride(0) != 1:
        raise ValueError("Original SG core requires unit-stride positions; forward copying is forbidden")
    key, value = state["key"][:, :num_ctx], state["value"][:, :num_ctx]
    if num_ctx == 0:
        return key, value
    normalized = state["normed"][:num_ctx]
    ops.rms_norm(normalized, context_states, model._hidden_norm_weight, model._rms_norm_eps)
    projected = state["projected"][:num_ctx]
    # Same existing all-layer unquantized projection, with an explicit stable
    # output. No second weight copy/transposed materialization is introduced.
    torch.mm(normalized, state["weight"].t(), out=projected)
    raw_kv = projected.view(num_ctx, state["layers"], 2 * state["heads"] * state["head_dim"])
    return _fused_norm_rope_stacked(raw_kv, state["norm_weight"], state["eps"],
        state["cos_sin"], context_positions, state["heads"], state["head_dim"],
        state["rotary_dim"], k_out=key, v_out=value)


def precompute_and_store(model, context_states, context_positions, context_slot_mapping=None):
    key, value = materialize(model, context_states, context_positions)
    if context_slot_mapping is None:
        return
    if context_states.shape[0] == 0:
        return
    v_scale = getattr(model.layers[0].self_attn, "v_scale", None)
    if v_scale is not None:
        value.mul_(v_scale)
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    if per_layer and len(context_slot_mapping) != model._num_attn_layers:
        raise ValueError("Per-layer cache slot mappings must cover all draft layers")
    # Preserve the actual backend's mapping, quantization/scale and null-slot
    # handling. No separate cache writer or memory layout is invented here.
    for index, attn in enumerate(model._attn_layers):
        slots = context_slot_mapping[index] if per_layer else context_slot_mapping
        if slots is not None:
            attn.impl.do_kv_cache_update(attn, key[index], value[index], attn.kv_cache, slots)


def status(model):
    state = getattr(model, "_dflash2_sg_kv", None)
    if state is None:
        return {"prepared": False}
    names = ("normed", "projected", "key", "value", "eps", "cos_sin")
    return {"prepared": True, "source_revision": SOURCE_REVISION, "capacity": state["capacity"],
        "scope": "SGLang FP32 K-norm+RoPE arithmetic; one serial worker; original vLLM cache writer",
        "tensors": {name: {"shape": list(state[name].shape), "dtype": str(state[name].dtype),
            "stride": list(state[name].stride()), "data_ptr": state[name].data_ptr(),
            "bytes": state[name].numel() * state[name].element_size()} for name in names},
        "workspace_bytes_excluding_shared_weights_and_rotary": sum(
            state[name].numel() * state[name].element_size() for name in ("normed", "projected", "key", "value", "eps")),
        "Graph_lifetime": "Model owns all buffers; drain/reset every graph before releasing the model."}


def is_supported(model, capacity):
    """Metadata-only admission; unsupported original DFlash keeps its own path."""
    if not HAS_TRITON:
        return False
    if tuple(getattr(model.config, "architectures", ()) or ()) != ("DFlash2DraftModel",):
        return False
    if not isinstance(capacity, int) or capacity <= 0:
        return False
    weight = getattr(model, "_fused_kv_weight", None)
    if weight is None or getattr(model, "_fused_kv_bias", None) is not None:
        return False
    return (
        weight.dtype == torch.bfloat16 and weight.device.type == "cuda"
        and tuple(weight.shape) == (10240, 2560)
        and model.config.hidden_size == 2560
        and (model._num_attn_layers, model._num_kv_heads, model._head_dim) == (5, 8, 128)
        and len(model._attn_layers) == 5 and model._rope_is_neox
        and model.layers[0].self_attn.rotary_emb.rotary_dim == 128
        and tuple(model._k_norm_weights.shape) == (5, 128)
    )


def prepare_if_supported(model, capacity):
    if is_supported(model, capacity):
        prepare(model, capacity)


def can_materialize(model, context_states, context_positions):
    """Use original math for unsupported runtime input without growing scratch."""
    state = getattr(model, "_dflash2_sg_kv", None)
    return (
        state is not None and context_states.ndim == 2
        and context_states.shape[1] == 2560
        and 0 <= context_states.shape[0] <= state["capacity"]
        and context_states.dtype == state["dtype"] and context_states.device == state["device"]
        and context_positions.ndim == 1 and context_positions.shape[0] == context_states.shape[0]
        and context_positions.dtype == torch.int64 and context_positions.device == state["device"]
        and context_positions.stride(0) == 1
    )
