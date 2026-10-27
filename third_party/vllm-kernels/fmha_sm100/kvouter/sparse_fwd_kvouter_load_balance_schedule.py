# Copyright (c) 2026 Fireworks AI
# SPDX-License-Identifier: Apache-2.0

"""Load-balancing scheduler for the sparse KV-outer attention forward.

Produces a fixed-size work-list that balances Q-token work across KV blocks, fully
on-device (no device->host copy). The number of selected KV blocks is data-dependent
and lives only on the GPU; this routine never reads it back to the host.

Scheme
------
Flatten all work into one global sequence of "work-units" (one (query, block) pair) ordered
by ``(kv_head, kv_block, query-within-block)``. Let ``total_work`` be its length (device only).
The grid is exactly ``num_splits``; with a fixed ``nqps`` (work-units per split), split ``s``
owns the contiguous range ``[s*nqps, min((s+1)*nqps, total_work))``. A split may span multiple
small blocks or be one chunk of a large block, so per-split work is balanced regardless of how
the selection is distributed. For each split we emit a **start** and an exclusive **end** tuple
``(kv_head, kv_block_idx, q_idx)`` (``q_idx`` = query position within that block, into
``kv_to_q_offsets``); splits past the work (``s*nqps >= total_work``) are ``(-1, -1, -1)``.

Single custom kernel (cuteDSL)
------------------------------
This replaces the former multi-op torch pipeline (diff + cumsum + arange + 2x searchsorted +
gather + where -> ~8 launches, materializing ``counts`` and ``cu`` over ``num_flat = Hkv*nbs``)
with ONE cuteDSL kernel and no intermediate tensors. Key fact: ``kv_to_q_offsets[h]`` is already
the per-head prefix sum, so the flattened prefix ``cu`` never needs materializing. The global
start of flat block ``(h, blk)`` is ``head_base[h] + (offsets[h, blk] - offsets[h, 0])`` where
``head_base[h] = sum_{h'<h} (offsets[h', nbs] - offsets[h', 0])``. One thread per split decodes
its two positions by: (1) an ``Hkv``-step scan to find the head (largest ``h`` with
``head_base[h] <= pos``) and ``total_work``; (2) a binary search in that head's offset row for
the block. Cost ``O(Hkv + log nbs)`` per split; integer-only; no device->host sync.

Sizing (host-only, no D2H)
--------------------------
Grid is exactly ``num_splits``. ``nqps = ceil(max_total_work / num_splits)`` where
``max_total_work = Hkv*total_q*topk`` is a host capacity bound (per kv-head each of ``total_q``
queries selects at most ``topk`` blocks). The real ``total_work`` never exceeds the bound, so
all real splits fit and the tail is sentinel.
"""

from __future__ import annotations

from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr

from flash_attn.cute.cute_dsl_utils import to_cute_tensor

__all__ = ["build_load_balanced_schedule"]


class _LoadBalanceScheduler:
    """Single-kernel device scheduler. ``hkv`` is compile-time (the Hkv scan is unrolled);
    ``nbs`` (block-slot count) and ``nqps`` are RUNTIME scalars, so this compiles once for any
    batch size / seqlen. The binary-search depth is a FIXED constant upper bound (the search is
    idempotent once converged), so it does not depend on ``nbs``."""

    # Fixed binary-search depth: covers nbs up to 2**_SEARCH_ITERS (>> any realistic B*msb).
    _SEARCH_ITERS = 32

    def __init__(self, hkv: int, num_splits: int, num_threads: int = 256):
        self.hkv = hkv
        self.num_splits = num_splits
        self.num_threads = num_threads
        self.search_iters = self._SEARCH_ITERS

    @cute.jit
    def __call__(self, mOffsets: cute.Tensor, mWorkStart: cute.Tensor, mWorkEnd: cute.Tensor,
                 nbs: Int32, nqps: Int64, stream=None):
        grid = (self.num_splits + self.num_threads - 1) // self.num_threads
        self.kernel(mOffsets, mWorkStart, mWorkEnd, nbs, nqps).launch(
            grid=[grid, 1, 1], block=[self.num_threads, 1, 1], stream=stream,
        )

    @cute.jit
    def _decode_write(self, mOffsets: cute.Tensor, mWork: cute.Tensor, s: Int32, pos: Int64,
                      nbs: Int32):
        """Decode flat position ``pos`` -> (kv_head, kv_block, q) and write mWork[s].

        ``pos`` is a flattened index into the global (head, block, q-within-block) work
        sequence. Valid work items use ``0 <= pos < total``; ``pos == total`` is the
        exclusive end (one past the last query in the last nonempty block). ``nbs`` is the
        runtime block-slot count (index of the per-head total in ``mOffsets``).
        """
        # (1) find head: largest h with head_base[h] <= pos (head_base monotone increasing).
        running = Int64(0)
        head = Int32(0)
        head_base = Int64(0)
        for h in cutlass.range_constexpr(self.hkv):
            base_h = running
            take = base_h <= pos
            head = Int32(h) if take else head
            head_base = base_h if take else head_base
            running += Int64(mOffsets[h, nbs]) - Int64(mOffsets[h, 0])
        local = pos - head_base
        total_h = Int64(mOffsets[head, nbs]) - Int64(mOffsets[head, 0])
        o0 = Int64(mOffsets[head, 0])
        blk = Int32(0)
        q = Int32(0)
        if local >= total_h:
            # Exclusive end at pos == total (or past sparse tail): find the last block with
            # any work. Logarithmic search with a fixed (idempotent) depth.
            lo = Int32(0)
            hi = nbs - Int32(1)
            for _ in cutlass.range_constexpr(self.search_iters):
                mid = (lo + hi + Int32(1)) >> Int32(1)
                take = Int64(mOffsets[head, mid]) < Int64(mOffsets[head, nbs])
                lo = mid if take else lo
                hi = hi if take else (mid - Int32(1))
            blk = lo
            q = Int32(mOffsets[head, blk + 1] - mOffsets[head, blk])
        else:
            # In-range: find blk with offsets[blk] <= local < offsets[blk+1].
            lo = Int32(0)
            hi = nbs - Int32(1)
            for _ in cutlass.range_constexpr(self.search_iters):
                mid = (lo + hi + Int32(1)) >> Int32(1)
                take = Int64(mOffsets[head, mid]) <= local
                lo = mid if take else lo
                hi = hi if take else (mid - Int32(1))
            blk = lo
            q = Int32(local - (Int64(mOffsets[head, blk]) - o0))
        mWork[s, 0] = head
        mWork[s, 1] = blk
        mWork[s, 2] = q

    @cute.kernel
    def kernel(self, mOffsets: cute.Tensor, mWorkStart: cute.Tensor, mWorkEnd: cute.Tensor,
               nbs: Int32, nqps: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        s = bidx * self.num_threads + tidx
        if s < Int32(self.num_splits):
            total = Int64(0)
            for h in cutlass.range_constexpr(self.hkv):
                total += Int64(mOffsets[h, nbs]) - Int64(mOffsets[h, 0])
            pos_start = Int64(s) * nqps
            if pos_start >= total:
                mWorkStart[s, 0] = Int32(-1); mWorkStart[s, 1] = Int32(-1); mWorkStart[s, 2] = Int32(-1)
                mWorkEnd[s, 0] = Int32(-1); mWorkEnd[s, 1] = Int32(-1); mWorkEnd[s, 2] = Int32(-1)
            else:
                pos_end = pos_start + nqps
                pos_end = pos_end if pos_end < total else total
                self._decode_write(mOffsets, mWorkStart, s, pos_start, nbs)
                self._decode_write(mOffsets, mWorkEnd, s, pos_end, nbs)


_compile_cache: dict = {}


def _get_compiled(hkv: int, num_splits: int, templates):
    # nbs is a runtime kernel arg (not a compile key), so the scheduler compiles once per
    # (hkv, num_splits) and is reused across all batch sizes / seqlens.
    key = (hkv, num_splits)
    if key not in _compile_cache:
        kernel = _LoadBalanceScheduler(hkv, num_splits)
        off_t, ws_t, we_t = templates
        _compile_cache[key] = cute.compile(
            kernel, off_t, ws_t, we_t, Int32(1), Int64(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _compile_cache[key]


def build_load_balanced_schedule(
    kv_to_q_offsets: torch.Tensor,  # [Hkv, num_block_slots + 1] int32, monotone per row
    *,
    total_q: int,
    num_splits: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build the balanced (start, end) work-list. Returns ``(work_start, work_end,
    grid_size, nqps)``: ``work_start``/``work_end`` are ``[grid_size, 3]`` int32 device
    tensors of ``(kv_head, kv_block_idx, q_idx)`` (end exclusive), sentinel ``(-1,-1,-1)``
    past the work. One cuteDSL kernel; no device->host sync."""
    assert kv_to_q_offsets.dim() == 2
    assert kv_to_q_offsets.dtype == torch.int32
    hkv, nbs1 = kv_to_q_offsets.shape
    nbs = nbs1 - 1
    device = kv_to_q_offsets.device

    # Host-only sizing (see module docstring): grid == num_splits, nqps vs the capacity bound.
    max_total_work = hkv * total_q * topk
    nqps = max(1, (max_total_work + num_splits - 1) // num_splits)
    grid_size = num_splits

    work_start = torch.empty(grid_size, 3, dtype=torch.int32, device=device)
    work_end = torch.empty(grid_size, 3, dtype=torch.int32, device=device)

    offs = kv_to_q_offsets.contiguous()
    off_t = to_cute_tensor(offs, assumed_align=4, leading_dim=1)
    ws_t = to_cute_tensor(work_start, assumed_align=4, leading_dim=1)
    we_t = to_cute_tensor(work_end, assumed_align=4, leading_dim=1)
    compiled = _get_compiled(int(hkv), int(num_splits), (off_t, ws_t, we_t))
    compiled(offs, work_start, work_end, Int32(nbs), Int64(nqps))
    return work_start, work_end, grid_size, nqps
