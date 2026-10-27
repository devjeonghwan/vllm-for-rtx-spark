# Copyright (c) 2026 Fireworks AI
# SPDX-License-Identifier: Apache-2.0

"""Build KV-stationary sparse index tensors from per-Q top-k block selection.

Inverts q→kv block selection into the KV-outer kernel contract:

  * ``topk_slot_ids``            ``[Hkv, num_block_slots * ratio]`` int64
  * ``kv_to_q_offsets``          ``[Hkv, num_block_slots + 1]`` int32 (monotone)
  * ``kv_to_q_indices_and_ranks`` ``[Hkv, Tq * topK, 2]`` int32 ``(q_index, rank)``

CuTe GPU pipeline:

1. **Init** ``topk_slot_ids`` and zero per-slot counts.
2. **Count** edges via atomic add into ``count`` + ``edge_local``.
3. **Offsets** via CuTe prefix-sum kernel on ``count`` → ``kv_to_q_offsets``.
4. **Scatter** ``(q, rank)`` into ``kv_to_q_indices_and_ranks``.

Slot id (merge key): ``slot = seq_id * msb + sparse_block_index`` per KV head.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr

# Count-phase atomic privatization: the edge-count atomic_add hammers a small set of
# slot counters (num_block_slots ~ KV blocks) from Tq*topk edges -> heavy contention
# (the dominant "other"-kernel cost; see sec 7an / 7ao). Spread it across R per-slot replica
# counters (count[hkv, slot, R], atomic into replica bidx%R), then ReduceReplicas (a
# block-per-slot smem scan: sum R for the slot total + exclusive prefix over R for each edge's
# base) restores the CSR order.
#
# R is ADAPTIVE to the contention level: contention ~= edges / (nbins*R), where edges =
# Tq*Hkv*topk and nbins = Hkv*num_block_slots. Few bins + many edges (short ctx / long Tq,
# e.g. tq16384 kv16384 = 128 bins, 262k edges) need a big R to break the atomic serialization;
# many bins (long ctx) are already low-contention and use the R=16 floor. ReduceReplicas cost
# is ~flat in R (block-per-slot), so a high R is free there. R is capped (smem/threads in the
# reduce + global memory R*nbins). MINIMAX_KERNELS_KVOUTER_COUNT_REPLICAS overrides (A/B; R=1 == original
# single-counter). Scales to 1M seqlen: at long ctx contention is low so R stays at the floor,
# keeping R*num_block_slots small in global memory.
_COUNT_REPLICAS_OVERRIDE = os.environ.get("MINIMAX_KERNELS_KVOUTER_COUNT_REPLICAS")
_COUNT_REPLICAS_FLOOR = 16
_COUNT_REPLICAS_MAX = 128
_COUNT_TARGET_CONTENTION = 4  # desired edges per replica-counter


def _adaptive_replicas(edges: int, nbins: int) -> int:
    """Pick R so atomic contention (edges per replica-counter) ~= _COUNT_TARGET_CONTENTION,
    clamped to [floor, max] and rounded up to a power of two. Env overrides for A/B."""
    if _COUNT_REPLICAS_OVERRIDE is not None:
        return max(1, int(_COUNT_REPLICAS_OVERRIDE))
    if nbins <= 0:
        return _COUNT_REPLICAS_FLOOR
    target = edges // (nbins * _COUNT_TARGET_CONTENTION)
    r = _COUNT_REPLICAS_FLOOR
    while r < target and r < _COUNT_REPLICAS_MAX:
        r <<= 1
    return max(_COUNT_REPLICAS_FLOOR, min(_COUNT_REPLICAS_MAX, r))


from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute import utils

__all__ = [
    "build_kvouter_index",
    "nested_selection_to_selected",
    "build_kvouter_index_from_nested",
    "q_to_seq_from_cu_seqlens",
]


def q_to_seq_from_cu_seqlens(
    cu_seqlens_q: Optional[torch.Tensor],
    total_q: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the ``[total_q] int32`` per-query sequence index from ``cu_seqlens_q`` ``[B+1]``.

    Convenience for callers of :func:`build_kvouter_index` / ``kvouter_attention`` (which require
    ``q_to_seq``). Single sequence (``cu_seqlens_q`` is ``None`` or has <=2 entries) -> all zeros.
    """
    if cu_seqlens_q is None or cu_seqlens_q.numel() <= 2:
        return torch.zeros(total_q, dtype=torch.int32, device=device)
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(q_lens.numel(), device=device, dtype=torch.int32),
        q_lens,
        output_size=total_q,
    )


class _InitSlotsAndCountsKernel:
    # num_block_slots / msb / block_table_cols are RUNTIME Int32 args (not constexpr), so this
    # kernel compiles once and serves any batch size / seqlen. See _build_kvouter_index_cute.
    def __init__(
        self,
        *,
        hkv: int,
        ratio: int,
        page_size: int,
        has_block_tables: bool,
        replicas: int = 1,
        num_threads: int = 512,
    ):
        self.hkv = hkv
        self.ratio = ratio
        self.page_size = page_size
        self.has_block_tables = has_block_tables
        self.replicas = replicas
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSelected: cute.Tensor,
        mBlockTables: cute.Tensor,
        mCount: cute.Tensor,
        mTopkSlotIds: cute.Tensor,
        num_block_slots: Int32,
        msb: Int32,
        block_table_cols: Int32,
        stream=None,
    ):
        self.kernel(
            mSelected,
            mBlockTables,
            mCount,
            mTopkSlotIds,
            num_block_slots,
            msb,
            block_table_cols,
        ).launch(
            grid=[self.hkv * num_block_slots, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSelected: cute.Tensor,
        mBlockTables: cute.Tensor,
        mCount: cute.Tensor,
        mTopkSlotIds: cute.Tensor,
        num_block_slots: Int32,
        msb: Int32,
        block_table_cols: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        h = Int32(bidx) // num_block_slots
        slot = Int32(bidx) - h * num_block_slots

        # Zero this slot's R replica counters (one lane each; num_threads >= replicas).
        if tidx < const_expr(self.replicas):
            mCount[h, slot, tidx] = Int32(0)
        if tidx == 0:
            for p in cutlass.range_constexpr(self.ratio):
                slot_col = slot * self.ratio + p
                slot_id = Int64(-1)
                if const_expr(self.has_block_tables):
                    seq_id = slot // msb
                    sb = slot - seq_id * msb
                    col = sb * self.ratio + p
                    if col < block_table_cols:
                        phys_page = Int32(mBlockTables[seq_id, col])
                        # Unallocated page slots are padded with -1. Remap to this row's
                        # first allocated physical page (block_tables[*,0]), not literal page
                        # 0: tables store physical cache indices and page 0 may belong to
                        # another sequence in the shared pool. Rows beyond used_kv_lens
                        # are masked in the count and forward kernels.
                        if phys_page < 0:
                            phys_page = Int32(mBlockTables[seq_id, 0])
                            if phys_page < 0:
                                phys_page = Int32(0)
                        if phys_page >= 0:
                            slot_id = Int64((phys_page * self.hkv + h) * self.page_size)
                else:
                    page = slot * self.ratio + p
                    slot_id = Int64((page * self.hkv + h) * self.page_size)
                mTopkSlotIds[h, slot_col] = slot_id


class _CountToOffsetsSerialKernel:
    """One thread per head: serial exclusive prefix sum (best for small ``num_block_slots``).
    ``num_block_slots`` is a RUNTIME Int32 arg, so this compiles once for any seqlen/batch."""

    def __init__(self, *, hkv: int, replicas: int = 1, num_threads: int = 128):
        self.hkv = hkv
        self.replicas = replicas
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mCount: cute.Tensor,
        mOffsets: cute.Tensor,
        mSelSlots: cute.Tensor,
        mSelOffsets: cute.Tensor,
        mNumSel: cute.Tensor,
        num_block_slots: Int32,
        stream=None,
    ):
        self.kernel(mCount, mOffsets, mSelSlots, mSelOffsets, mNumSel, num_block_slots).launch(
            grid=[self.hkv, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCount: cute.Tensor,
        mOffsets: cute.Tensor,
        mSelSlots: cute.Tensor,
        mSelOffsets: cute.Tensor,
        mNumSel: cute.Tensor,
        num_block_slots: Int32,
    ):
        # Dense exclusive prefix sum + FUSED selected-slot compaction in the same pass: emit
        # (sel_slots[j]=slot, sel_offsets[j]=exclusive prefix) for each count>0 slot. This avoids
        # a separate compaction kernel launch (which regressed small/dense shapes). Only
        # sel_slots[0, num_sel) is written (the tail is unused); sel_offsets plateaus at the head
        # total for j>=num_sel.
        tidx, _, _ = cute.arch.thread_idx()
        h, _, _ = cute.arch.block_idx()
        if tidx == 0:
            running = Int32(0)
            mOffsets[h, 0] = running
            j = Int32(0)
            for slot in cutlass.range(num_block_slots, unroll=1):
                c = Int32(mCount[h, slot])
                if c > 0:
                    mSelSlots[h, j] = slot
                    mSelOffsets[h, j] = running  # exclusive prefix == dense offset[slot]
                    j += 1
                running += c
                mOffsets[h, slot + 1] = running
            mNumSel[h] = j
            for jj in cutlass.range(num_block_slots + 1, unroll=1):
                if jj >= j:
                    mSelOffsets[h, jj] = running  # plateau at total -> sel_offsets[nbs]==total


class _CountToOffsetsParallelKernel:
    """256-thread chunked parallel prefix sum (best for large ``num_block_slots``).
    ``num_block_slots`` / ``chunk_size`` are RUNTIME Int32 args (smem is a fixed 256*2 ints
    independent of them), so this compiles once for any seqlen/batch."""

    _NUM_THREADS = 256

    def __init__(self, *, hkv: int, replicas: int = 1):
        self.hkv = hkv
        self.replicas = replicas
        self.num_threads = self._NUM_THREADS
        self.smem_ints = self._NUM_THREADS * 4  # sChunk, sBase (dense) + sSelChunk, sSelBase (compact)

    @cute.jit
    def __call__(
        self,
        mCount: cute.Tensor,
        mOffsets: cute.Tensor,
        mSelSlots: cute.Tensor,
        mSelOffsets: cute.Tensor,
        mNumSel: cute.Tensor,
        num_block_slots: Int32,
        chunk_size: Int32,
        stream=None,
    ):
        self.kernel(mCount, mOffsets, mSelSlots, mSelOffsets, mNumSel, num_block_slots, chunk_size).launch(
            grid=[self.hkv, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self.smem_ints * 4 + 256,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCount: cute.Tensor,
        mOffsets: cute.Tensor,
        mSelSlots: cute.Tensor,
        mSelOffsets: cute.Tensor,
        mNumSel: cute.Tensor,
        num_block_slots: Int32,
        chunk_size: Int32,
    ):
        # 256-thread chunked dense prefix sum + FUSED selected-slot compaction (avoids a separate
        # compaction kernel launch). A second parallel prefix over the count>0 indicator gives each
        # selected slot its compact index j; sel_offsets is plateau-filled with the head total then
        # the scatter overwrites [0, num_sel). Only sel_slots[0, num_sel) is written (tail unused).
        tidx, _, _ = cute.arch.thread_idx()
        h, _, _ = cute.arch.block_idx()
        start = Int32(tidx) * chunk_size

        smem = cutlass.utils.SmemAllocator()
        sChunk = smem.allocate_tensor(Int32, cute.make_layout(self._NUM_THREADS), byte_alignment=16)
        sBase = smem.allocate_tensor(Int32, cute.make_layout(self._NUM_THREADS), byte_alignment=16)
        sSelChunk = smem.allocate_tensor(Int32, cute.make_layout(self._NUM_THREADS), byte_alignment=16)
        sSelBase = smem.allocate_tensor(Int32, cute.make_layout(self._NUM_THREADS), byte_alignment=16)

        # Phase 1: per-lane sum of counts + per-lane count of selected (count>0) slots.
        lane_sum = Int32(0)
        lane_sel = Int32(0)
        for i in cutlass.range(chunk_size, unroll=1):
            slot = start + i
            if slot < num_block_slots:
                c = Int32(mCount[h, slot])
                if c > 0:
                    lane_sel += 1
                lane_sum += c
        sChunk[tidx] = lane_sum
        sSelChunk[tidx] = lane_sel
        cute.arch.barrier()

        # Phase 2: thread 0 builds both exclusive prefixes; num_sel = sum of selected counts.
        if tidx == 0:
            mOffsets[h, 0] = Int32(0)
            running = Int32(0)
            sBase[0] = Int32(0)
            sel_running = Int32(0)
            sSelBase[0] = Int32(0)
            for i in cutlass.range(1, self._NUM_THREADS, unroll=1):
                running += sChunk[i - 1]
                sBase[i] = running
                sel_running += sSelChunk[i - 1]
                sSelBase[i] = sel_running
            mNumSel[h] = sel_running + sSelChunk[self._NUM_THREADS - 1]
        cute.arch.barrier()

        # total == dense prefix over all lanes; available to every thread after the barrier.
        total = sBase[self._NUM_THREADS - 1] + sChunk[self._NUM_THREADS - 1]

        # Phase 2.5: plateau-fill sel_offsets[0..nbs] = total (chunked over compact-j). The scatter
        # below overwrites [0, num_sel); the tail stays at total so sel_offsets[nbs]==total.
        for i in cutlass.range(chunk_size, unroll=1):
            j = start + i
            if j <= num_block_slots:
                mSelOffsets[h, j] = total
        cute.arch.barrier()

        # Phase 3: dense offsets write + compact scatter at j = sel base + local selected rank.
        running = sBase[tidx]
        sel_j = sSelBase[tidx]
        for i in cutlass.range(chunk_size, unroll=1):
            slot = start + i
            if slot < num_block_slots:
                c = Int32(mCount[h, slot])
                if c > 0:
                    mSelSlots[h, sel_j] = slot
                    mSelOffsets[h, sel_j] = running  # exclusive prefix == dense offset[slot]
                    sel_j += 1
                running += c
                mOffsets[h, slot + 1] = running


_COUNT_TO_OFFSETS_PARALLEL_THRESHOLD = 128


def _make_count_to_offsets_kernel(*, hkv: int, parallel: bool, replicas: int = 1):
    # `parallel` is decided host-side from num_block_slots; each variant compiles once
    # (num_block_slots is a runtime kernel arg).
    if parallel:
        return _CountToOffsetsParallelKernel(hkv=hkv, replicas=replicas)
    return _CountToOffsetsSerialKernel(hkv=hkv, replicas=replicas)


class _ReduceReplicasKernel:
    """Reduce the per-slot R replica counters -> slot total, and overwrite each replica with
    its exclusive prefix (the replica's base offset within the slot, consumed by ScatterRanks).

    ONE BLOCK per (hkv, slot) with a warp-wide (or R-wide) smem Hillis-Steele scan over the R
    replicas. Block-per-slot is essential when R is large: at high contention we want a big R
    to spread the CountEdges atomics, but the reduce then has R-way work per slot. A
    thread-per-slot serial R-loop collapses to a single under-occupied CTA at small
    num_block_slots (e.g. 128 slots at tq16384) and balloons to ~40us at R=256; spreading
    slots across CTAs (each scanning R in smem) keeps it ~flat in R. num_units / num_block_slots
    are RUNTIME Int32 args; R is constexpr (smem layout + scan steps), so this compiles once per
    R for any seqlen/batch."""

    def __init__(self, *, replicas: int):
        self.replicas = replicas
        # One warp minimum; >=R threads so each replica is owned by one lane (coalesced load).
        self.num_threads = max(32, ((replicas + 31) // 32) * 32)

    @cute.jit
    def __call__(
        self, mCount: cute.Tensor, mTotal: cute.Tensor, num_units: Int32, num_block_slots: Int32, stream=None
    ):
        self.kernel(mCount, mTotal, num_units, num_block_slots).launch(
            grid=[num_units, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self.replicas * 4 + 256,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mCount: cute.Tensor, mTotal: cute.Tensor, num_units: Int32, num_block_slots: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        # grid == num_units, so unit = bidx is always in range.
        unit = Int32(bidx)
        h = unit // num_block_slots
        slot = unit - h * num_block_slots
        R = const_expr(self.replicas)

        if const_expr(R == 1):
            # Trivial: single replica -> base 0, total = its count.
            if tidx == 0:
                c = Int32(mCount[h, slot, 0])
                mCount[h, slot, 0] = Int32(0)
                mTotal[h, slot] = c
            return

        smem = cutlass.utils.SmemAllocator()
        s = smem.allocate_tensor(Int32, cute.make_layout(R), byte_alignment=16)
        own = Int32(0)
        if tidx < R:
            own = Int32(mCount[h, slot, tidx])
            s[tidx] = own
        cute.arch.barrier()
        # Inclusive Hillis-Steele scan over the R replicas (log2(R) steps, R constexpr).
        d = 1
        while d < R:
            v = Int32(0)
            if tidx < R and tidx >= d:
                v = Int32(s[tidx - d])
            cute.arch.barrier()
            if tidx < R:
                s[tidx] = Int32(s[tidx]) + v
            cute.arch.barrier()
            d *= 2
        if tidx < R:
            # exclusive prefix = inclusive - self (the replica's base within the slot).
            mCount[h, slot, tidx] = Int32(s[tidx]) - own
        if tidx == 0:
            mTotal[h, slot] = Int32(s[R - 1])


_reduce_replicas_compile_cache: dict = {}


class _CountEdgesKernel:
    # cap (=tq*h_idx*topk), msb and n_batches are RUNTIME Int32 args; the per-query sequence id is
    # found in-kernel by a binary search over cu_seqlens_q (mCuSeqlensQ), so no [Tq] q_to_seq tensor
    # is materialized. Compiles once for any tq/batch/seqlen.
    def __init__(
        self,
        *,
        h_idx: int,
        topk: int,
        block_size: int,
        causal: bool,
        has_block_tables: bool,
        ratio: int,
        replicas: int = 1,
        num_threads: int = 256,
    ):
        self.h_idx = h_idx
        self.topk = topk
        self.block_size = block_size
        self.causal = causal
        self.has_block_tables = has_block_tables
        self.ratio = ratio
        self.replicas = replicas
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSelected: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mTopkSlotIds: cute.Tensor,
        mCount: cute.Tensor,
        mEdgeLocal: cute.Tensor,
        cap: Int32,
        msb: Int32,
        n_batches: Int32,
        stream=None,
    ):
        self.kernel(
            mSelected,
            mCuSeqlensQ,
            mSeqUsedK,
            mTopkSlotIds,
            mCount,
            mEdgeLocal,
            cap,
            msb,
            n_batches,
        ).launch(
            grid=[(cap + self.num_threads - 1) // self.num_threads, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSelected: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mTopkSlotIds: cute.Tensor,
        mCount: cute.Tensor,
        mEdgeLocal: cute.Tensor,
        cap: Int32,
        msb: Int32,
        n_batches: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        edge = bidx * self.num_threads + tidx
        if edge < cap:
            t = edge // (self.h_idx * self.topk)
            rem = edge - t * (self.h_idx * self.topk)
            h_i = rem // self.topk
            rank = rem - h_i * self.topk
            sb = Int32(mSelected[t, h_i, rank])
            valid = sb >= 0 and sb < msb

            # Per-query sequence id via binary search over cu_seqlens_q [B+1] (monotone): the
            # largest seq_id with cu_seqlens_q[seq_id] <= t. Replaces a materialized [Tq] q_to_seq
            # lookup. n_batches == 1 -> seq_id 0 (the while never iterates).
            seq_lo = Int32(0)
            seq_hi = n_batches - Int32(1)
            while seq_lo < seq_hi:
                seq_mid = (seq_lo + seq_hi + Int32(1)) >> Int32(1)
                seq_take = Int64(mCuSeqlensQ[seq_mid]) <= Int64(t)
                seq_lo = seq_mid if seq_take else seq_lo
                seq_hi = seq_hi if seq_take else (seq_mid - Int32(1))
            seq_id = seq_lo

            # Causal block cap: derive the query's absolute position from cu_seqlens_q +
            # used_kv_lens (right-aligned suffix) instead of a [Tq] positions tensor:
            #   pos = (t - q_off) + (Lk_b - tq_b);  keep block sb iff sb <= pos // block_size.
            if const_expr(self.causal):
                if valid:
                    q_off = Int32(mCuSeqlensQ[seq_id])
                    tq_b = Int32(mCuSeqlensQ[seq_id + 1]) - q_off
                    lkb = Int32(mSeqUsedK[seq_id])
                    pos = (t - q_off) + (lkb - tq_b)
                    max_sb = pos // self.block_size + 1
                    valid = sb < max_sb

            # selected num heads == hkv: index head maps directly to its KV head.
            h_kv = Int32(h_i)

            edge_slot = seq_id * msb + sb
            if valid:
                if const_expr(self.has_block_tables):
                    for p in cutlass.range_constexpr(self.ratio):
                        if mTopkSlotIds[h_kv, edge_slot * self.ratio + p] < Int64(0):
                            valid = False
                if valid:
                    # Privatized count: atomic into replica (bidx % R) of this slot. Returns
                    # the within-replica rank; the global base is added in ScatterRanks from
                    # the replica prefix CountToOffsets writes back into mCount.
                    r = Int32(bidx) % const_expr(self.replicas)
                    local = cute.arch.atomic_add(
                        ptr=utils.elem_pointer(mCount, (h_kv, edge_slot, r)).llvm_ptr,
                        val=Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    mEdgeLocal[edge] = local


class _ScatterRanksKernel:
    # cap (=tq*h_idx*topk), msb and n_batches are RUNTIME Int32 args; the per-query sequence id is
    # found in-kernel by a binary search over cu_seqlens_q (mCuSeqlensQ), so no [Tq] q_to_seq tensor
    # is materialized. Compiles once for any tq/batch/seqlen.
    def __init__(
        self,
        *,
        h_idx: int,
        topk: int,
        block_size: int,
        causal: bool,
        has_block_tables: bool,
        ratio: int,
        replicas: int = 1,
        num_threads: int = 256,
    ):
        self.h_idx = h_idx
        self.topk = topk
        self.block_size = block_size
        self.causal = causal
        self.replicas = replicas
        self.has_block_tables = has_block_tables
        self.ratio = ratio
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mEdgeLocal: cute.Tensor,
        mSelected: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mTopkSlotIds: cute.Tensor,
        mOffsets: cute.Tensor,
        mCount: cute.Tensor,
        mIdxRanks: cute.Tensor,
        mInv: cute.Tensor,
        cap: Int32,
        msb: Int32,
        n_batches: Int32,
        stream=None,
    ):
        self.kernel(
            mEdgeLocal,
            mSelected,
            mCuSeqlensQ,
            mSeqUsedK,
            mTopkSlotIds,
            mOffsets,
            mCount,
            mIdxRanks,
            mInv,
            cap,
            msb,
            n_batches,
        ).launch(
            grid=[(cap + self.num_threads - 1) // self.num_threads, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mEdgeLocal: cute.Tensor,
        mSelected: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mTopkSlotIds: cute.Tensor,
        mOffsets: cute.Tensor,
        mCount: cute.Tensor,
        mIdxRanks: cute.Tensor,
        mInv: cute.Tensor,
        cap: Int32,
        msb: Int32,
        n_batches: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        edge = bidx * self.num_threads + tidx
        if edge < cap:
            t = edge // (self.h_idx * self.topk)
            rem = edge - t * (self.h_idx * self.topk)
            h_i = rem // self.topk
            rank = rem - h_i * self.topk
            sb = Int32(mSelected[t, h_i, rank])
            valid = sb >= 0 and sb < msb

            # Per-query sequence id via binary search over cu_seqlens_q [B+1] (monotone): the
            # largest seq_id with cu_seqlens_q[seq_id] <= t. Replaces a materialized [Tq] q_to_seq
            # lookup. n_batches == 1 -> seq_id 0 (the while never iterates).
            seq_lo = Int32(0)
            seq_hi = n_batches - Int32(1)
            while seq_lo < seq_hi:
                seq_mid = (seq_lo + seq_hi + Int32(1)) >> Int32(1)
                seq_take = Int64(mCuSeqlensQ[seq_mid]) <= Int64(t)
                seq_lo = seq_mid if seq_take else seq_lo
                seq_hi = seq_hi if seq_take else (seq_mid - Int32(1))
            seq_id = seq_lo

            # Causal block cap: derive the query's absolute position from cu_seqlens_q +
            # used_kv_lens (right-aligned suffix) instead of a [Tq] positions tensor:
            #   pos = (t - q_off) + (Lk_b - tq_b);  keep block sb iff sb <= pos // block_size.
            if const_expr(self.causal):
                if valid:
                    q_off = Int32(mCuSeqlensQ[seq_id])
                    tq_b = Int32(mCuSeqlensQ[seq_id + 1]) - q_off
                    lkb = Int32(mSeqUsedK[seq_id])
                    pos = (t - q_off) + (lkb - tq_b)
                    max_sb = pos // self.block_size + 1
                    valid = sb < max_sb

            # selected num heads == hkv: index head maps directly to its KV head.
            h_kv = Int32(h_i)

            edge_slot = seq_id * msb + sb
            # Inverse map (tile-ordered combine gather): every (t, h, rank) edge is visited
            # exactly once, so inv is written UNCONDITIONALLY (-1 for dropped edges) -- the
            # tensor needs no init fill. `out` is exactly the pair position p this edge's
            # partial will occupy in the flat O/stats stream.
            inv_val = Int32(-1)
            if valid:
                if const_expr(self.has_block_tables):
                    for p in cutlass.range_constexpr(self.ratio):
                        if mTopkSlotIds[h_kv, edge_slot * self.ratio + p] < Int64(0):
                            valid = False
                if valid:
                    # CSR position = slot start + replica base (exclusive prefix over R, written
                    # back into mCount by CountToOffsets) + within-replica rank (mEdgeLocal).
                    r = Int32(bidx) % const_expr(self.replicas)
                    local = Int32(mEdgeLocal[edge]) + Int32(mCount[h_kv, edge_slot, r])
                    out = Int32(mOffsets[h_kv, edge_slot]) + local
                    mIdxRanks[h_kv, out, 0] = Int32(t)
                    mIdxRanks[h_kv, out, 1] = Int32(rank)
                    inv_val = out
            mInv[h_kv, t, rank] = inv_val


_init_slots_compile_cache: dict = {}
_count_edges_compile_cache: dict = {}
_offsets_compile_cache: dict = {}
_scatter_compile_cache: dict = {}


def _dummy_block_tables(selected: torch.Tensor) -> torch.Tensor:
    return selected[0]


def _build_kvouter_index_cute(
    selected: torch.Tensor,
    *,
    hkv: int,
    topk: int,
    num_block_slots: int,
    block_size: int,
    page_size: int,
    cu_seqlens_q: torch.Tensor,
    causal: bool,
    used_kv_lens: torch.Tensor,
    block_tables: Optional[torch.Tensor],
    msb: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tq, h_idx, _ = selected.shape
    device = selected.device
    ratio = block_size // page_size
    max_pairs = tq * h_idx * topk
    # Per-request shape scalars passed to the kernels as RUNTIME Int32 args (not constexpr), so
    # every kernel below compiles once and is reused across all batch sizes / seqlens. n_batches
    # (= B) bounds the in-kernel binary search over cu_seqlens_q that yields the per-query seq id.
    cap = tq * h_idx * topk
    n_batches = cu_seqlens_q.shape[0] - 1
    block_table_cols = 0 if block_tables is None else int(block_tables.shape[1])
    parallel_offsets = num_block_slots > _COUNT_TO_OFFSETS_PARALLEL_THRESHOLD
    nt_off = _CountToOffsetsParallelKernel._NUM_THREADS
    # +1 so the fused-compaction plateau loop covers the compact-j endpoint nbs (sel_offsets[nbs]).
    chunk_size = (num_block_slots + 1 + nt_off - 1) // nt_off

    replicas = _adaptive_replicas(cap, hkv * num_block_slots)
    # Per-slot count is privatized across R replica counters to break atomic contention in
    # CountEdges (sec 7an); CountToOffsets reduces R -> slot total and overwrites each replica
    # with its exclusive prefix (replica base) for the scatter. R=1 == single-counter original.
    count = torch.empty(hkv, num_block_slots, replicas, dtype=torch.int32, device=device)
    # Per-slot total (sum over replicas), produced by the reduce kernel and prefix-summed into
    # offsets. Separate from `count` (which becomes the per-replica exclusive prefix base).
    count_total = torch.empty(hkv, num_block_slots, dtype=torch.int32, device=device)
    edge_local = torch.empty(tq * h_idx * topk, dtype=torch.int32, device=device)
    topk_slot_ids = torch.empty(hkv, num_block_slots * ratio, dtype=torch.int64, device=device)
    kv_to_q_offsets = torch.empty(hkv, num_block_slots + 1, dtype=torch.int32, device=device)
    # Compact selected-slot index, produced FUSED inside CountToOffsets (no separate kernel
    # launch): sel_slots[j] = j-th selected slot, sel_offsets = compact CSR plateaued at the head
    # total, num_sel = selected slots per head. The kernel only scatters sel_slots[0, num_sel); the
    # tail [num_sel, nbs) is left UNINITIALIZED (the scheduler + forward iterate only [0, num_sel),
    # bounded by num_sel), so no -1 fill is needed -- skipping it avoids a per-call fill launch.
    sel_slots = torch.empty(hkv, num_block_slots, dtype=torch.int32, device=device)
    sel_offsets = torch.empty(hkv, num_block_slots + 1, dtype=torch.int32, device=device)
    num_sel = torch.empty(hkv, dtype=torch.int32, device=device)
    kv_to_q_indices_and_ranks = torch.empty(hkv, max_pairs, 2, dtype=torch.int32, device=device)
    # Inverse map inv[hkv, q, rank] -> pair position (-1 = dropped edge); written
    # unconditionally by the scatter kernel, so no init fill.
    inv = torch.empty(hkv, tq, topk, dtype=torch.int32, device=device)

    seqused_k_arg = used_kv_lens.to(device=device, dtype=torch.int32).contiguous()
    block_tables_arg = block_tables if block_tables is not None else _dummy_block_tables(selected)
    if block_tables is not None:
        block_tables_arg = block_tables_arg.contiguous()

    selected_t = to_cute_tensor(selected, assumed_align=4, leading_dim=2)
    cuq_t = to_cute_tensor(cu_seqlens_q, assumed_align=8, leading_dim=0)
    sk_t = to_cute_tensor(seqused_k_arg, assumed_align=4, leading_dim=0)
    block_tables_t = to_cute_tensor(block_tables_arg, assumed_align=4, leading_dim=1)
    count_t = to_cute_tensor(count, assumed_align=4, leading_dim=2)
    count_total_t = to_cute_tensor(count_total, assumed_align=4, leading_dim=1)
    edge_local_t = to_cute_tensor(edge_local, assumed_align=4, leading_dim=0)
    slot_t = to_cute_tensor(topk_slot_ids, assumed_align=8, leading_dim=1)
    off_t = to_cute_tensor(kv_to_q_offsets, assumed_align=4, leading_dim=1)
    sel_slots_t = to_cute_tensor(sel_slots, assumed_align=4, leading_dim=1)
    sel_offsets_t = to_cute_tensor(sel_offsets, assumed_align=4, leading_dim=1)
    num_sel_t = to_cute_tensor(num_sel, assumed_align=4, leading_dim=0)
    ir_t = to_cute_tensor(kv_to_q_indices_and_ranks, assumed_align=4, leading_dim=2)
    inv_t = to_cute_tensor(inv, assumed_align=4, leading_dim=2)

    init_key = (hkv, ratio, page_size, block_tables is not None, replicas)
    if init_key not in _init_slots_compile_cache:
        _init_slots_compile_cache[init_key] = cute.compile(
            _InitSlotsAndCountsKernel(
                hkv=hkv,
                ratio=ratio,
                page_size=page_size,
                has_block_tables=block_tables is not None,
                replicas=replicas,
            ),
            selected_t,
            block_tables_t,
            count_t,
            slot_t,
            num_block_slots,
            msb,
            block_table_cols,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _init_slots_compile_cache[init_key](
        selected,
        block_tables_arg,
        count,
        topk_slot_ids,
        num_block_slots,
        msb,
        block_table_cols,
    )

    count_key = (
        h_idx,
        topk,
        block_size,
        causal,
        block_tables is not None,
        ratio,
        replicas,
    )
    if count_key not in _count_edges_compile_cache:
        _count_edges_compile_cache[count_key] = cute.compile(
            _CountEdgesKernel(
                h_idx=h_idx,
                topk=topk,
                block_size=block_size,
                causal=causal,
                has_block_tables=block_tables is not None,
                ratio=ratio,
                replicas=replicas,
            ),
            selected_t,
            cuq_t,
            sk_t,
            slot_t,
            count_t,
            edge_local_t,
            cap,
            msb,
            n_batches,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _count_edges_compile_cache[count_key](
        selected,
        cu_seqlens_q,
        seqused_k_arg,
        topk_slot_ids,
        count,
        edge_local,
        cap,
        msb,
        n_batches,
    )

    # Reduce R replica counters -> per-slot total (+ overwrite replicas with exclusive prefix
    # for the scatter), parallel over slots. Always run (R=1 is a trivial pass-through).
    num_units = hkv * num_block_slots
    reduce_key = (replicas,)
    if reduce_key not in _reduce_replicas_compile_cache:
        _reduce_replicas_compile_cache[reduce_key] = cute.compile(
            _ReduceReplicasKernel(replicas=replicas),
            count_t,
            count_total_t,
            Int32(num_units),
            num_block_slots,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _reduce_replicas_compile_cache[reduce_key](count, count_total, num_units, num_block_slots)

    # CountToOffsets does the dense prefix sum AND fuses the selected-slot compaction (sel_slots/
    # sel_offsets/num_sel) in the same launch, so there is no separate compaction kernel.
    offsets_key = (hkv, parallel_offsets)
    if offsets_key not in _offsets_compile_cache:
        if parallel_offsets:
            _offsets_compile_cache[offsets_key] = cute.compile(
                _make_count_to_offsets_kernel(hkv=hkv, parallel=True),
                count_total_t,
                off_t,
                sel_slots_t,
                sel_offsets_t,
                num_sel_t,
                num_block_slots,
                chunk_size,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        else:
            _offsets_compile_cache[offsets_key] = cute.compile(
                _make_count_to_offsets_kernel(hkv=hkv, parallel=False),
                count_total_t,
                off_t,
                sel_slots_t,
                sel_offsets_t,
                num_sel_t,
                num_block_slots,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
    if parallel_offsets:
        _offsets_compile_cache[offsets_key](
            count_total, kv_to_q_offsets, sel_slots, sel_offsets, num_sel, num_block_slots, chunk_size
        )
    else:
        _offsets_compile_cache[offsets_key](
            count_total, kv_to_q_offsets, sel_slots, sel_offsets, num_sel, num_block_slots
        )

    scatter_key = (
        h_idx,
        topk,
        block_size,
        causal,
        block_tables is not None,
        ratio,
        replicas,
    )
    if scatter_key not in _scatter_compile_cache:
        _scatter_compile_cache[scatter_key] = cute.compile(
            _ScatterRanksKernel(
                h_idx=h_idx,
                topk=topk,
                block_size=block_size,
                causal=causal,
                has_block_tables=block_tables is not None,
                ratio=ratio,
                replicas=replicas,
            ),
            edge_local_t,
            selected_t,
            cuq_t,
            sk_t,
            slot_t,
            off_t,
            count_t,
            ir_t,
            inv_t,
            cap,
            msb,
            n_batches,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _scatter_compile_cache[scatter_key](
        edge_local,
        selected,
        cu_seqlens_q,
        seqused_k_arg,
        topk_slot_ids,
        kv_to_q_offsets,
        count,
        kv_to_q_indices_and_ranks,
        inv,
        cap,
        msb,
        n_batches,
    )

    return topk_slot_ids, kv_to_q_offsets, kv_to_q_indices_and_ranks, inv, sel_slots, sel_offsets, num_sel


def build_kvouter_index(
    selected: torch.Tensor,
    *,
    hkv: int,
    topk: int,
    num_block_slots: int,
    block_size: int = 128,
    page_size: int = 64,
    block_tables: torch.Tensor,
    msb: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    causal: bool = False,
    used_kv_lens: Optional[torch.Tensor] = None,
) -> Tuple:
    """Invert ``selected [Tq, Hkv, topK]`` into KV-outer index tensors; also returns the
    inverse map ``inv [Hkv, Tq, topK] -> pair position`` (-1 = dropped edge) consumed by the
    tile-ordered combine gather.

    Returns the 7-tuple ``(topk_slot_ids, kv_to_q_offsets, kv_to_q_indices_and_ranks, inv,
    sel_slots, sel_offsets, num_sel)``: the dense index plus the compact selected-slot index
    (always produced fused inside CountToOffsets) that the scheduler + forward iterate.

    The per-query sequence id is found in-kernel by a binary search over ``cu_seqlens_q``
    (``[B+1]`` monotone), so no ``[Tq]`` ``q_to_seq`` tensor is materialized.

    Args:
        selected: sparse block indices per Q / index-head / rank; ``-1`` pads.
        hkv: number of KV heads on this rank.
        topk: top-k width (must match ``selected.shape[2]``).
        num_block_slots: fat block-slot table width per KV head (``B * msb``).
        block_size: sparse KV block size in tokens (128).
        page_size: paged KV page size (64 or 128).
        block_tables: REQUIRED ``[B, M]`` physical page ids per sequence.
        msb: REQUIRED sparse blocks per sequence (``= block_tables.shape[1] // ratio``); the merge
            slot is ``seq_id * msb + sparse_block``, so it must be the PER-SEQUENCE block count
            (not ``num_block_slots = B * msb``).
        cu_seqlens_q: ``[B+1]`` cumulative query lengths for batched varlen (cast to int64 here);
            ``None`` (single sequence) is treated as ``[0, Tq]``.
        causal: if True, drop ``(query, block)`` edges past the query's causal range. The query's
            absolute position is derived in-kernel from ``cu_seqlens_q`` + ``used_kv_lens`` (right-
            aligned suffix), so no ``[Tq]`` positions tensor is needed.
        used_kv_lens: ``[B]`` real per-sequence KV length ``Lk_b``. Required semantics under
            ``causal`` (supports variable / non-128-multiple lengths); ``None`` defaults to the
            uniform ``msb * block_size`` (legacy assumption). Ignored when ``causal=False``.
    """
    assert selected.ndim == 3 and selected.dtype == torch.int32
    tq, h_idx, topk_sel = selected.shape
    assert topk_sel == topk
    assert h_idx == hkv, f"selected num heads ({h_idx}) must equal hkv ({hkv}): one block selection per KV head"
    assert block_tables is not None, "block_tables is required"
    assert msb is not None, "msb is required (per-sequence sparse block count)"
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor([0, tq], dtype=torch.int64, device=selected.device)
    else:
        cu_seqlens_q = cu_seqlens_q.to(device=selected.device, dtype=torch.int64).contiguous()
    ratio = block_size // page_size
    assert block_size % page_size == 0

    n_batches = cu_seqlens_q.shape[0] - 1
    # The init/count/scatter kernels index block_tables[seq_id] and used_kv_lens[seq_id] with
    # seq_id in [0, n_batches) (init: seq_id = slot // msb, slot < num_block_slots = B*msb), so
    # both must cover every sequence or the kernels read out of bounds / mis-map pages.
    assert (
        block_tables.shape[0] >= n_batches
    ), f"block_tables must have >= n_batches={n_batches} rows, got {block_tables.shape[0]}"
    if used_kv_lens is None:
        used_kv_lens = torch.full((n_batches,), msb * block_size, dtype=torch.int32, device=selected.device)
    else:
        assert (
            used_kv_lens.shape[0] == n_batches
        ), f"used_kv_lens length ({used_kv_lens.shape[0]}) must equal n_batches ({n_batches})"

    result = _build_kvouter_index_cute(
        selected,
        hkv=hkv,
        topk=topk,
        num_block_slots=num_block_slots,
        block_size=block_size,
        page_size=page_size,
        cu_seqlens_q=cu_seqlens_q,
        causal=causal,
        used_kv_lens=used_kv_lens,
        block_tables=block_tables,
        msb=msb,
    )
    return result


def nested_selection_to_selected(
    selection: list,
    *,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert ``selection[h_kv][q] -> [blocks]`` to ``[Tq, Hkv, topK]`` int32."""
    hkv = len(selection)
    tq = len(selection[0])
    out = torch.full((tq, hkv, topk), -1, dtype=torch.int32, device=device)
    for h in range(hkv):
        for q in range(tq):
            blks = selection[h][q]
            for r, b in enumerate(blks[:topk]):
                out[q, h, r] = int(b)
    return out


def build_kvouter_index_from_nested(
    selection: list,
    *,
    topk: int,
    num_block_slots: int,
    block_size: int = 128,
    page_size: int = 64,
    causal: bool = False,
    used_kv_lens: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple:
    """Build index tensors from nested test-style ``selection`` lists.

    Single sequence: ``cu_seqlens_q`` defaults to ``[0, Tq]`` (every query maps to sequence 0)
    and an identity ``block_tables`` (``arange``) is synthesized so the contiguous timeline maps
    page ``p`` -> physical page ``p`` (``msb = num_block_slots``). ``causal`` / ``used_kv_lens``
    are forwarded (``used_kv_lens`` defaults to the uniform ``num_block_slots * block_size``).
    """
    hkv = len(selection)
    if device is None:
        device = used_kv_lens.device if used_kv_lens is not None else torch.device("cuda")
    selected = nested_selection_to_selected(selection, topk=topk, device=device)
    ratio = block_size // page_size
    block_tables = torch.arange(num_block_slots * ratio, dtype=torch.int32, device=device).view(1, -1)
    return build_kvouter_index(
        selected,
        hkv=hkv,
        topk=topk,
        num_block_slots=num_block_slots,
        block_tables=block_tables,
        msb=num_block_slots,
        block_size=block_size,
        page_size=page_size,
        causal=causal,
        used_kv_lens=used_kv_lens,
    )
