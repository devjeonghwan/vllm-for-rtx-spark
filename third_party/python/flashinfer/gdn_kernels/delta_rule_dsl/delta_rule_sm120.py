"""Original FI host API with the original compiled ARM64 C-host getter."""
from __future__ import annotations
from types import SimpleNamespace
import torch
from ...utils import get_device_sm_count, _get_cache_buf
from .varlen_helper import is_integer_dtype
from original_gdn_aot import get_prefill_kernel as _get_prefill_kernel, cached_native as get_cached_compile

# Explicit dtype metadata keys only: no CuTe interpreter/compiler is emulated.
cutlass = SimpleNamespace(Float16="original_Float16", BFloat16="original_BFloat16",
    Float32="original_Float32", Float8E4M3FN="original_Float8E4M3FN", Float8E5M2="original_Float8E5M2")

def _sm120_compile_options(device):
    return ("original_ARM64_C_ABI_AOT",)

def state_dtype_to_cutlass(dtype: torch.dtype) -> type[cutlass.Numeric]:
    state_dtypes = {
        torch.float32: cutlass.Float32,
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e5m2: cutlass.Float8E5M2,
    }
    if dtype not in state_dtypes:
        raise ValueError(
            f"Unsupported state dtype {dtype}, expected float32, bfloat16, "
            "float16, float8_e4m3fn, or float8_e5m2"
        )
    return state_dtypes[dtype]

class _FullyFusedDeltaRuleSm120:
    @staticmethod
    def can_implement(
            num_q_heads: int,
            num_k_heads: int,
            num_v_heads: int,
            head_size: int,
            element_size: int,
        ) -> bool:
            ratio = (
                num_q_heads // num_v_heads
                if num_q_heads > num_v_heads
                else num_v_heads // num_q_heads
            )
            is_gva_enabled = num_v_heads > num_q_heads

            is_gqa_like = (
                (num_k_heads == num_v_heads)
                and (num_q_heads == ratio * num_k_heads)
                and (num_q_heads == ratio * num_v_heads)
            )
            is_gva_like = (
                (num_q_heads == num_k_heads)
                and (num_v_heads == ratio * num_q_heads)
                and (num_v_heads == ratio * num_k_heads)
            )

            alignment = 16 // element_size
            return (
                ((not is_gva_enabled and is_gqa_like) or (is_gva_enabled and is_gva_like))
                and (head_size <= 128)
                and ((head_size % alignment) == 0)
            )

def delta_rule_prefill_dsl(
    o: torch.Tensor,  # (total_seqlen, num_o_heads, D) fp16/bf16, output
    state: torch.Tensor,  # (num_seqs, num_sab_heads, D, D) fp32, output
    q: torch.Tensor,  # (total_seqlen, num_q_heads, D)
    k: torch.Tensor,  # (total_seqlen, num_k_heads, D)
    v: torch.Tensor,  # (total_seqlen, num_v_heads, D)
    init_state: torch.Tensor | None,  # (num_seqs, num_sab_heads, D, D) fp32, optional
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    cu_seqlens: torch.Tensor,  # (num_seqs+1,) int64
    scale: float,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    state_indices: torch.Tensor | None = None,
):

    device = q.device
    D = q.shape[-1]

    num_seqs = cu_seqlens.shape[0] - 1
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)

    if not _FullyFusedDeltaRuleSm120.can_implement(
        num_q_heads, num_k_heads, num_v_heads, D, q.element_size()
    ):
        raise RuntimeError("can_implement failed")
    if D != 128:
        raise RuntimeError(f"DSL kernel only supports D=128, got {D}")

    needs_alpha = alpha is not None
    needs_beta = beta is not None
    needs_init_state = init_state is not None
    needs_checkpointing = checkpoint_every_n_tokens > 0
    use_state_indices = state_indices is not None
    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }.get(q.dtype)
    if kernel_dtype is None:
        raise RuntimeError(f"DSL kernel only supports fp16/bf16 inputs, got {q.dtype}")

    if k.dtype != q.dtype or v.dtype != q.dtype or o.dtype != q.dtype:
        raise RuntimeError(
            f"q/k/v/o dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}, {o.dtype}"
        )
    if alpha is not None and alpha.dtype != torch.float32:
        raise RuntimeError(f"alpha must have dtype torch.float32, got {alpha.dtype}")
    if beta is not None and beta.dtype != torch.float32:
        raise RuntimeError(f"beta must have dtype torch.float32, got {beta.dtype}")
    if init_state is not None:
        state_dtype_to_cutlass(init_state.dtype)
    state_dtype_to_cutlass(state.dtype)
    if state_checkpoints is not None:
        state_dtype_to_cutlass(state_checkpoints.dtype)
    if not is_integer_dtype(cu_seqlens.dtype):
        raise RuntimeError(
            f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
        )

    expected_state_tail = (num_sab_heads, D, D)
    for name, tensor in (("state", state), ("init_state", init_state)):
        if tensor is None:
            continue
        if (
            not use_state_indices and tensor.shape != (num_seqs, *expected_state_tail)
        ) or (use_state_indices and tuple(tensor.shape[1:]) != expected_state_tail):
            raise RuntimeError(
                f"{name} must have shape "
                f"{('[N_pool]' if use_state_indices else f'[{num_seqs}]')} + {expected_state_tail}, "
                f"got {tuple(tensor.shape)}"
            )
    if use_state_indices and (
        not is_integer_dtype(state_indices.dtype) or state_indices.shape != (num_seqs,)
    ):
        raise RuntimeError(
            f"state_indices must have shape {(num_seqs,)} and an integer dtype"
        )

    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("o", o),
        ("cu_seqlens", cu_seqlens),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (
        ("alpha", alpha),
        ("beta", beta),
        ("state_indices", state_indices),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (("state", state), ("init_state", init_state)):
        if tensor is None:
            continue
        if not use_state_indices and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    total_seqlen = q.shape[0]
    num_o_heads = o.shape[1]
    q_tma = q.as_strided(
        (total_seqlen, D, num_q_heads),
        (num_q_heads * D, 1, D),
    )
    k_tma = k.as_strided(
        (D, total_seqlen, num_k_heads),
        (1, num_k_heads * D, D),
    )
    v_tma = v.as_strided(
        (D, total_seqlen, num_v_heads),
        (1, num_v_heads * D, D),
    )
    o_tma = o.as_strided(
        (D, total_seqlen, num_o_heads),
        (1, num_o_heads * D, D),
    )
    total_checkpoints = state_checkpoints.shape[0] if needs_checkpointing else 1

    workspace_size = get_device_sm_count(device) * 128
    tensormaps_t = _get_cache_buf("gdn_prefill_tensormaps", workspace_size, device)
    stream = torch.cuda.current_stream(device).cuda_stream

    delta_rule_kernel = _get_prefill_kernel(
        needs_alpha,
        needs_beta,
        needs_init_state,
        needs_checkpointing,
        kernel_dtype,
        initial_state_dtype=(init_state.dtype if needs_init_state else torch.float32),
        state_dtype=state.dtype,
        checkpoint_state_dtype=(
            state_checkpoints.dtype if needs_checkpointing else torch.float32
        ),
        use_state_indices=use_state_indices,
        cu_seqlens_dtype=cu_seqlens.dtype,
        state_indices_dtype=state_indices.dtype if use_state_indices else None,
        checkpoint_cu_starts_dtype=(
            checkpoint_cu_starts.dtype if needs_checkpointing else None
        ),
        state_inner_strides=(tuple(state.stride()[1:]) if use_state_indices else None),
        init_state_inner_strides=(
            tuple(init_state.stride()[1:])
            if use_state_indices and needs_init_state
            else None
        ),
    )

    compile_options = _sm120_compile_options(device)
    compiled_delta_rule_kernel = get_cached_compile(delta_rule_kernel, compile_options)
    if compiled_delta_rule_kernel is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        kernel_args = (
            from_dlpack(q_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(o_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            (
                from_dlpack(alpha.reshape(-1), assumed_align=16).mark_layout_dynamic()
                if needs_alpha
                else None
            ),
            (
                from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic()
                if needs_beta
                else None
            ),
            from_dlpack(state, assumed_align=16).mark_layout_dynamic(),
            (
                from_dlpack(init_state, assumed_align=16).mark_layout_dynamic()
                if needs_init_state
                else None
            ),
            (
                from_dlpack(state_indices, assumed_align=4).mark_layout_dynamic()
                if use_state_indices
                else None
            ),
            (
                from_dlpack(
                    state_checkpoints.reshape(-1), assumed_align=16
                ).mark_layout_dynamic()
                if needs_checkpointing
                else None
            ),
            (
                from_dlpack(checkpoint_cu_starts, assumed_align=8).mark_layout_dynamic()
                if needs_checkpointing
                else None
            ),
            from_dlpack(tensormaps_t, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Float32(scale),
            cutlass.Int32(num_q_heads),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_v_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(num_seqs),
            cutlass.Int32(total_checkpoints),
            cutlass.Int32(checkpoint_every_n_tokens),
            num_seqs * num_sab_heads,
            stream,
        )
        compiled_delta_rule_kernel = cached_compile(
            delta_rule_kernel,
            *kernel_args,
            compile_options=compile_options,
        )
    compiled_delta_rule_kernel(
        q_tma,
        k_tma,
        v_tma,
        o_tma,
        alpha.reshape(-1) if needs_alpha else None,
        beta.reshape(-1) if needs_beta else None,
        state,
        init_state if needs_init_state else None,
        state_indices if use_state_indices else None,
        state_checkpoints.reshape(-1) if needs_checkpointing else None,
        checkpoint_cu_starts if needs_checkpointing else None,
        tensormaps_t,
        cu_seqlens,
        scale,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        num_sab_heads,
        num_seqs,
        total_checkpoints,
        checkpoint_every_n_tokens,
        num_seqs * num_sab_heads,
        stream,
    )
