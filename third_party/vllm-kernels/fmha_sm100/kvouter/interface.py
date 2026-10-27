# Copyright (c) 2026 Fireworks AI
# SPDX-License-Identifier: Apache-2.0

"""Public interface for MiniMax M3 KV-outer sparse attention.

The public entry point builds the KV-stationary index, runs the sparse forward
kernel, and log-sum-exp merges rank partials. CuTe-DSL dependencies are imported
lazily so importing this package does not initialize CUDA.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

HEAD_DIM = 128
BLOCK_SIZE = 128
DEFAULT_PAGE_SIZE = 64

_SUPPORTED_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)

__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "can_run_sparse_kvouter",
    "kvouter_attention",
]


_FA4_DEPS_ERROR: Optional[BaseException] = None


def _check_fa4_deps() -> bool:
    """Return whether the CuTe-DSL dependencies used by these kernels resolve."""
    global _FA4_DEPS_ERROR
    try:
        # lazy: CuTe-DSL imports initialize compiler/runtime state.
        import cutlass.cute  # noqa: F401
        import flash_attn.cute.flash_fwd_sm100  # noqa: F401
        from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned  # noqa: F401
    except Exception as exc:
        _FA4_DEPS_ERROR = exc
        return False
    return True


def _require_fa4_deps() -> None:
    """Raise a focused error when the required public CuTe packages are absent
    or incompatible (the underlying import error is chained)."""
    if not _check_fa4_deps():
        raise ImportError(
            "MiniMax M3 KV-outer sparse attention requires nvidia-cutlass-dsl, "
            "quack-kernels and the FlashAttention-4 CuTe package "
            "(flash-attn-4==4.0.0b15, which imports only with nvidia-cutlass-dsl "
            f"4.5.x). Underlying error: {_FA4_DEPS_ERROR!r}"
        ) from _FA4_DEPS_ERROR


def can_run_sparse_kvouter(dtype: Optional[torch.dtype] = None) -> bool:
    """Capability + dependency check: Blackwell (SM100+) and ``flash_attn.cute`` deps."""
    if dtype is not None and dtype not in _SUPPORTED_DTYPES:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return False
    if major < 10:
        return False
    return _check_fa4_deps()


def _varlen_meta(
    q: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    block_size: int,
    page_size: int,
) -> Tuple[int, int, int, Optional[torch.Tensor]]:
    """Derive paging/varlen scalars for the KV-outer API.

    Returns ``(batch_size, msb, num_block_slots, cu_seqlens_q_i64)``.
    When supplied, ``cu_seqlens_q`` is cast to contiguous int64 once here. KV-outer requires it;
    pass the explicit B=1 form ``[0, Tq]`` for a single sequence.
    """
    ratio = block_size // page_size
    msb = max(1, block_tables.shape[1] // ratio)
    batch_size = 1 if cu_seqlens_q is None else cu_seqlens_q.shape[0] - 1
    num_block_slots = msb if batch_size == 1 else batch_size * msb
    if cu_seqlens_q is None:
        return batch_size, msb, num_block_slots, None
    cu_seqlens_q_i64 = cu_seqlens_q.to(device=q.device, dtype=torch.int64).contiguous()
    return batch_size, msb, num_block_slots, cu_seqlens_q_i64


def kvouter_attention(
    q: torch.Tensor,  # [Tq, Hq, D] token-major
    k_cache: torch.Tensor,  # [num_pages, Hkv, page_size, D]
    v_cache: torch.Tensor,
    selected: torch.Tensor,  # [Tq, Hkv, topK] int32 block ids (-1 padded)
    block_tables: torch.Tensor,  # [B, max_blocks] paged block table
    *,
    cu_seqlens_q: torch.Tensor,  # [B+1] cumulative query lengths (REQUIRED; [0, Tq] for single seq)
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    used_kv_lens: Optional[torch.Tensor] = None,  # [B] real per-seq KV length Lk_b
    block_size: int = BLOCK_SIZE,
    page_size: int = DEFAULT_PAGE_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KV-outer (split-KV / KV-stationary) sparse attention — index build + forward + merge.

    Inverts the per-query top-K ``selected`` into the KV-stationary index, runs the forward
    (each KV block is loaded once and scores the GQA-packed ``(query, qhead)`` rows that selected
    it, emitting one fp32 partial per ``(q_token, rank)``), then log-sum-exp-merges the partials
    into the final output.

    Args:
        q: Queries, token-major ``[Tq, Hq, D]`` (head dim ``D == 128``), dtype bf16, fp16, or fp8.
            ``Tq`` is the total query-token count (summed over batches for varlen).
        k_cache: Paged key cache, ``[num_pages, Hkv, page_size, D]``, same dtype as ``q``. The
            number of KV heads ``Hkv`` is read from ``k_cache.shape[1]`` (GQA group ``Hq // Hkv``).
        v_cache: Paged value cache, ``[num_pages, Hkv, page_size, D]``, same dtype as ``q``.
        selected: Per-query selected KV block ids, ``[Tq, Hkv, topK]`` int32; ``-1`` pads unused
            ranks. ``topK`` is read from ``selected.shape[-1]``.
        block_tables: Paged block table, ``[B, max_blocks]`` int32 — logical→physical page ids per
            sequence (``B`` = batch size; ``B == 1`` for a single sequence).
        cu_seqlens_q: REQUIRED ``[B+1]`` cumulative query lengths (cast to int64 once inside this
            function); pass ``[0, Tq]`` for a single sequence. The per-query sequence index is
            found in-kernel by a binary search over this (no ``[Tq]`` ``q_to_seq`` tensor needed).
        softmax_scale: QK softmax scale (Python ``float``). Default ``1/sqrt(D)``.
        causal: If True, apply causal masking; the per-block column limit is computed in-kernel
            from ``cu_seqlens_q`` + ``used_kv_lens`` (right-aligned suffix), so no ``[Tq]``
            positions tensor is needed.
        used_kv_lens: ``[B]`` int32 real per-sequence KV length ``Lk_b`` (supports variable /
            non-128-multiple lengths). Drives both causal (suffix limit) and non-causal (padding)
            masking. ``None`` defaults to the uniform ``msb * block_size`` (legacy assumption);
            pass the real per-seq lengths (e.g. ``varlen.kv_seq_lens``) for varlen KV.
        block_size: Sparse KV block size in tokens (default 128).
        page_size: Paged-cache page size, 64 or 128 (default 64).
        out_dtype: dtype of the returned ``o`` (default ``torch.bfloat16``).
        return_lse: If True, also return the log-sum-exp; otherwise the second tuple element is ``None``.
        out: Optional preallocated output ``[Tq, Hq, D]`` of ``out_dtype`` (contiguous, on
            ``q.device``); written in place and returned as ``o``.

    Returns:
        Tuple ``(o, lse)``:
          * ``o``: attention output, token-major ``[Tq, Hq, D]``, dtype ``out_dtype``.
          * ``lse``: log-sum-exp, head-major ``[Hq, Tq]`` fp32 if ``return_lse`` else ``None``
            (the FlashAttention forward convention).
    """
    _require_fa4_deps()
    # lazy: CuTe-DSL kernel modules are loaded only for execution.
    from .build_kvouter_index import build_kvouter_index
    # lazy: CuTe-DSL kernel modules are loaded only for execution.
    from .sparse_fwd_kvouter import (
        sparse_kvouter_attn_fwd_indexed,
    )

    assert cu_seqlens_q is not None, "cu_seqlens_q is required for kvouter_attention; use [0, Tq]"

    # Backend selection is environment-controlled. The AOT C++ path is preferred
    # when the package extension is available; Python CuTe-DSL remains the fallback.
    env = os.environ.get("FMHA_SM100_KVOUTER_CPP")
    if env is None:
        env = os.environ.get("MINIMAX_KERNELS_KVOUTER_CPP")
    if env == "0":
        backend = "python"
    elif env == "1":
        backend = "cpp"
    else:
        # lazy: the optional extension and AOT exporter are only needed for C++ dispatch.
        from .cpp_backend import cpp_backend_available

        backend = "cpp" if cpp_backend_available() else "python"
    if backend == "cpp":
        # lazy: the optional extension and AOT exporter are only needed for C++ dispatch.
        from .cpp_backend import kvouter_attention_cpp

        return kvouter_attention_cpp(
            q,
            k_cache,
            v_cache,
            selected,
            block_tables,
            cu_seqlens_q=cu_seqlens_q,
            softmax_scale=softmax_scale,
            causal=causal,
            used_kv_lens=used_kv_lens,
            block_size=block_size,
            page_size=page_size,
            out_dtype=out_dtype,
            return_lse=return_lse,
            out=out,
        )
    assert backend == "python", f"unknown kvouter_attention backend {backend!r}"
    num_kv_heads = k_cache.shape[1]
    topk = selected.shape[-1]
    _, msb, num_block_slots, cu_seqlens_q_i64 = _varlen_meta(q, block_tables, cu_seqlens_q, block_size, page_size)
    slot_ids, offs, idx_ranks, inv, sel_slots, sel_offsets, num_sel = build_kvouter_index(
        selected.contiguous(),
        hkv=num_kv_heads,
        topk=topk,
        num_block_slots=num_block_slots,
        block_size=block_size,
        page_size=page_size,
        cu_seqlens_q=cu_seqlens_q_i64,
        causal=causal,
        used_kv_lens=used_kv_lens,
        block_tables=block_tables,
        msb=msb,
    )
    return sparse_kvouter_attn_fwd_indexed(
        q,
        k_cache,
        v_cache,
        slot_ids,
        offs,
        idx_ranks,
        topk=topk,
        block_size=block_size,
        page_size=page_size,
        softmax_scale=softmax_scale,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q_i64,
        used_kv_lens=used_kv_lens,
        out_dtype=out_dtype,
        return_lse=return_lse,
        out=out,
        inv=inv,
        sel_slots=sel_slots,
        sel_offsets=sel_offsets,
        num_sel=num_sel,
    )
