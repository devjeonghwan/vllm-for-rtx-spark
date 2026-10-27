# Portions adapted from FlashAttention-4's SM100 forward kernel:
# https://github.com/Dao-AILab/flash-attention/blob/6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15/flash_attn/cute/flash_fwd_sm100.py
#
# FlashAttention-4 is Copyright (c) 2022, the respective contributors, as
# shown by its AUTHORS file. All rights reserved.
#
# This implementation inherits or forks FlashAttentionForwardSm100 and its KV-load,
# online-softmax, correction-epilogue, pipeline, barrier, and tile-scheduler primitives.
# Fireworks' modifications replace dense Q-major traversal with MiniMax M3's
# KV-stationary sparse index, load-balanced scheduling, packed-GQA row gathering,
# per-rank partial output scattering, and paged-cache masking.
#
# Copyright (c) 2026 Fireworks AI
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

"""Blackwell (SM100) sparse KV-outer forward — GQA-packed, one tile per work-item.

Each persistent work-item is one **128-row tile of packed (query, qhead) rows** drawn
from a single ``(hkv, kv_block)``. A block selected by ``count`` queries has
``count*qhead`` virtual rows (numbered head-fastest, ``virtual = query*qhead + head``,
so the qhead divide is by a compile-time constant); these are split into
``ceil(count*qhead/128)`` tiles. The tile loads the block's 128 keys once and does ONE
QK over the full block, ONE softmax, ONE PV — no sub-block / qhead / q-tile loops.

This packs the whole GQA group (and multiple queries) into the MMA's M dimension, so a
single K load serves up to 128 rows of compute. Versus a (block, qhead) work-item it
both shares K/V across the group AND collapses the per-tile pipeline overhead that
dominates low-reuse (sparse) workloads — e.g. one 16-row tile instead of sixteen 1-row
tiles. Work-items == total tiles keeps SM occupancy high even when few blocks are
selected (the block's queries fan out across tiles/CTAs).

Reuses FA4 per-tile primitives (``softmax_step``, ``correction_epilogue``) and pipeline
classes. The gather/scatter/mask paths decode each row's ``(query, head, rank)``.
Outputs per-(q, rank) partials (default ``O_partial`` dtype matches ``q`` for bandwidth;
pass ``partial_dtype=torch.float32`` for tighter correctness checks):
  TILE-ORDERED ``O_partial [Hkv*Tq*topK*qhead, D]`` + stats ``(m~, l) [Hkv, Tq*topK*qhead]``
  by pair position p, plus the inverse map ``inv [Hkv, Tq, topK] -> p`` for the combine.
"""

import math
import os
from functools import lru_cache, partial
from typing import Optional, Tuple

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl.cutlass import CuTeDSL
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cutlass_dsl import const
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from quack import copy_utils
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor
from flash_attn.cute import utils
import flash_attn.cute.pipeline as pipeline_custom
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
    StaticPersistentTileScheduler,
)
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

from .sparse_fwd_kvouter_load_balance_schedule import (
    build_load_balanced_schedule,
)

__all__ = ["indexed_block_partials"]


class SparseKVOuterForward(FlashAttentionForwardSm100):
    """GQA-packed sparse KV-outer forward (SM100). Forks FA4's FlashAttentionForwardSm100
    for its per-tile primitives (load_KV / softmax_step / correction_epilogue) and
    pipeline classes, but owns its data movement: each work-item is a q_stage*128-row
    block of packed (query, qhead) rows over one resident 128-key block."""

    def __init__(
        self,
        qhead_per_kvhead: int,
        nheads_kv: int,
        page_size: int,
        causal: bool = False,
        q_load_stage: int = None,
        o_buffers: int = 2,
    ):
        # Single arch (store-in-correction): correction issues the O bulk-TMA itself (rolling
        # per-thread wait_group(o_buffers-1) drains -- producer and consumer of the sO ring are
        # the same warpgroup, so the sO mbarrier pipeline is bypassed), and the freed store warp
        # joins the two load warps for a cooperative 3-warp cp.async Q gather. A dedicated
        # store-warp variant was slower for M3 top-k selections.
        # q_load_stage decouples the Q smem pipeline DEPTH (load->mma prefetch) from the fixed
        # q_stage=2 softmax ping-pong. The Q-buffer index advances once per 128-row tile and
        # is independent of the tmem/softmax stage, so q_load_stage sets how far the load warps
        # run ahead of the MMA (hides the Q gather latency). Must be >= 2 so both in-flight
        # ping-pong tiles have a live buffer; default = 2 (no extra prefetch). Larger costs sQ
        # smem (q_dtype * 128*128 per stage).
        self._q_load_stage_cfg = q_load_stage if q_load_stage is not None else 2
        self._o_buffers_cfg = o_buffers
        # n_block_size = 128: the whole 128-key block is resident and consumed in one gemm.
        super().__init__(
            head_dim=128,
            head_dim_v=128,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            is_split_kv=False,
            pack_gqa=False,
            m_block_size=128,
            n_block_size=128,
            q_stage=2,
            is_persistent=True,
            paged_kv_non_tma=False,
            is_varlen_q=False,
            use_2cta_instrs=False,
            use_clc_scheduler=False,
        )
        self.nheads_kv = nheads_kv
        self.page_size = page_size
        self.ratio = 128 // page_size
        self.block_size = 128
        assert page_size in (64, 128), "page_size must be 64 or 128"
        self.use_block_sparsity = False
        # Causal is computed entirely in-kernel (see softmax_loop) from cu_seqlens_q +
        # uniform msb (= num_block_slots // n_batches); single-seq is the B=1 case.
        self.sparse_causal = causal
        # Compact selected-slot iteration (the only path): the work-item flat index is a COMPACT
        # index ``cj = head*nbs + j`` over only the SELECTED slots (sel_slots/sel_offsets/num_sel
        # from the fused CountToOffsets, threaded via self._mSelSlots/self._mNumSel set in mainloop),
        # so the forward visits exactly the selected blocks -- skipping both the msb-padding tail
        # AND unselected real blocks in one step (no dense gap-jump walk over real blocks).
        self.split_P_arrive = 0  # no split-P signaling in this simpler mainloop
        self.mask_mod = None  # not the mask_mod hook; see _apply_mask
        self.s0_s1_barrier = False  # independent softmax warpgroups (per-stage pipelines)
        self.use_tma_Q = False  # Q is gathered (cp.async), not TMA'd
        self.clc_scheduler_warp_id = None

        # Warp layout (no TMA on Q/O). Override FA4's default since use_tma_Q=False.
        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        # q_stage=2 ping-pong. A work-item is a 256-row block of packed (query, qhead) rows;
        # the two 128-row halves run concurrently on two softmax warpgroups over
        # double-buffered tStS/tOtO. One correction warpgroup handles both halves' rescale/
        # evac + LSE; the bulk-TMA O_partial scatter is OFFLOADED to a dedicated store warp
        # (only o_nbox lanes issue TMA boxes, so one warp suffices), taking the store issue
        # + drain off the correction critical path. Q is loaded via TMA ([qhead, K_ATOM]
        # K-half boxes into the MMA-A swizzle; see prototype/bench_tma_q_load.py); G2S TMA
        # issue is one-lane-per-warp, so the two load warps split the Q tiles by stage
        # parity (warp 13: K/V + stage-0 tiles, warp 14: stage-1 tiles) for issue throughput.
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.load_warp_ids = (13, 14)
        self.store_warp_id = 15
        self.empty_warp_ids = ()
        self.threads_per_cta = cute.arch.WARP_SIZE * 16

        # Per-warp register budget for the warp-specialized setmaxregister reallocation.
        # Owned here (not inherited from FlashAttentionForwardSm100) and tuned for this kernel's
        # warp layout. The CTA launches at 128 regs/thread (512 threads = the full 64K-register
        # SM file); at runtime setmaxregister redistributes among the 16 warps: the softmax
        # warpgroup(s) grow to num_regs_softmax, correction shrinks to num_regs_correction, and
        # load/mma/empty shrink to num_regs_other. Budget identity (4 warpgroup quotas of 128):
        #     num_regs_other == 512 - 2 * num_regs_softmax - num_regs_correction.
        # NB: the reallocation only takes effect because the kernel launches with
        # min_blocks_per_mp=1 (see __call__) — without it ptxas drops setmaxnreg and spills hard.
        # 176/96/64 was swept on B200: minimal spill (8 B/thread) and ~10% faster best/mid than
        # the inherited 192/80/48 (40 B spill).
        self.num_regs_softmax = 176
        self.num_regs_correction = 96
        self.num_regs_other = 512 - 2 * self.num_regs_softmax - self.num_regs_correction  # 64

    @cute.jit
    def _apply_mask(
        self,
        acc_S,
        n_block,
        thr_mma_qk,
        thr_tmem_load,
        row_start,
        n_rows,
        sPairs_slot,
        col_limit_base,
    ):
        """Right-aligned causal mask for the packed (query, qhead) rows of one tile.

        Row r (tile coord) is virtual row ``row_start + r``; ``query_local = virtual //
        qhead`` indexes the block's gathered query list (head only changes which Q rows
        attend, not the mask). Keep column ``col`` (0..127, position within the 128-key
        block, absolute key position ``sb*128 + col``) iff ``col <= limit``:

        * causal: ``limit = qidx + col_limit_base`` with ``col_limit_base = (Lk_b - Tq_b)
          - q_offset_b - sb*128``; i.e. key position <= query position ``(t - q_off) +
          (Lk_b - Tq_b)`` (right-aligned suffix). Uses the real per-seq KV length ``Lk_b``.
        * non-causal: ``limit = col_limit_base = (Lk_b - 1) - sb*128``; i.e. key position
          < ``Lk_b`` (drops the partial last block's padding columns).

        ``Lk_b`` (= ``used_kv_lens[b]``) makes both correct for variable per-seq KV
        lengths (and non-128-multiple lengths)."""
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        tScS = thr_mma_qk.partition_C(cS)[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        ncol = const_expr(cute.size(tScS_t2r.shape))
        qhead = const_expr(self.qhead_per_kvhead)
        # Per-thread column limit (each thread's fragment is exactly ONE row):
        #   causal:     limit = qidx + col_limit_base -- qidx from the load warp's sQIdxRank
        #               smem ring (valid post S-wait via the Q-full -> QK -> S-full chain),
        #               ONE smem read hoisted out of the element loop.
        #   non-causal: limit = col_limit_base (the Lk_b key-padding cut; uniform).
        # Fast path: if the limit keeps the whole 128-key block (limit >= n_block-1) the
        # mask is a no-op and the thread skips the compare/select loop entirely. Non-causal
        # skips uniformly except on the sequence's partial LAST block; causal skips at warp
        # granularity (a warp spans 2 boxes) for blocks fully in the query's past -- the
        # common case. OOB tail boxes carry INT32_MAX-ish sentinels (_q_src_rowgroups), so
        # their rows skip too (their outputs are store-dropped anyway); rows beyond n_rows
        # stay unmasked as before.
        if const_expr(self.sparse_causal):
            row = tScS_t2r[0][0]
            limit = Int32(sPairs_slot[row // qhead, 0] + col_limit_base)
        else:
            limit = Int32(col_limit_base)
        if limit < const_expr(self.n_block_size - 1):
            for i in cutlass.range_constexpr(ncol):
                if tScS_t2r[i][0] < n_rows:
                    acc_S[i] = acc_S[i] if (tScS_t2r[i][1] <= limit) else -Float32.inf

    @cute.jit
    def _decode_workitem(self, wi, mWorkStart, mWorkEnd, nbs):
        """Load-balanced work-item ``wi``: a contiguous run of the global (kv_head, kv_block,
        query) work sequence, given by the scheduler as ``[start, end)`` tuples
        ``(kv_head, kv_block, q_idx)`` (end exclusive). Flatten kv_head/kv_block into one
        block index ``fb = kv_head*nbs + kv_block`` so the run spans ``fb in [fb_s, fb_e]``;
        the run may cover many small blocks or a slice of a large one. Sentinel work-items
        (start kv_head == -1, past the real work) decode to ``valid=False`` / ``num_fb=0``."""
        hkv_s = Int32(mWorkStart[wi, 0])
        kvb_s = Int32(mWorkStart[wi, 1])
        q_s = Int32(mWorkStart[wi, 2])
        hkv_e = Int32(mWorkEnd[wi, 0])
        kvb_e = Int32(mWorkEnd[wi, 1])
        q_e = Int32(mWorkEnd[wi, 2])
        valid = hkv_s >= 0
        fb_s = hkv_s * nbs + kvb_s
        fb_e = hkv_e * nbs + kvb_e
        num_fb = Int32(0)
        if valid:
            num_fb = fb_e - fb_s + 1
        return valid, fb_s, fb_e, q_s, q_e, num_fb

    @cute.jit
    def _next_cj(self, cj, nbs):
        """Advance the compact index ``cj = head*nbs + j`` to the next selected slot, skipping the
        tail ``[num_sel[head], nbs)`` of the current head (head IS the segment in compact space;
        stride ``nbs``; per-head ``real = num_sel[head]``). Reads ``self._mNumSel`` (set in
        mainloop) to avoid threading it through every warp fn."""
        nxt = cj + 1
        h = nxt // nbs
        local = nxt - h * nbs
        real = Int32(self._mNumSel[h])
        if local >= real:
            nxt = (h + 1) * nbs
        return nxt

    @cute.jit
    def _num_real_cj(self, cj_s, cj_e, nbs):
        """Count selected slots in the compact run ``[cj_s, cj_e]`` (loop trip count). Sums, per
        spanned head, the overlap of ``[cj_s, cj_e]`` with that head's selected range
        ``[head*nbs, head*nbs + num_sel[head])``."""
        seg_s = cj_s // nbs
        seg_e = cj_e // nbs
        n = Int32(0)
        for s in cutlass.range(seg_e - seg_s + 1, unroll=1):
            h = seg_s + s
            base = h * nbs
            real = Int32(self._mNumSel[h])
            lo = base
            if lo < cj_s:
                lo = cj_s
            hi = base + real
            if hi > cj_e + 1:
                hi = cj_e + 1
            if hi > lo:
                n = n + (hi - lo)
        return n

    @cute.jit
    def _slice_block(self, fb, is_first, fb_e, q_s, q_e, nbs, mKvToQOffsets):
        """The (hkv, kv_block) and query slice this work-item processes for flat-block ``fb``.
        ``is_first`` (``fb == fb_s``) marks the run's first block, which starts at ``q_s`` (the rest
        at 0); the last block (``fb == fb_e``) ends at the exclusive ``q_e`` (the rest at the block's
        ``count``). ``base`` is folded to ``offset + q_lo`` so the gather/scatter/mask index
        the slice's queries directly; ``total_rows = (q_hi-q_lo)*qhead`` packed rows split into
        ``n_groups`` ping-pong groups. Empty slices (incl. unselected block-slots) -> n_groups==0."""
        hkv = fb // nbs
        kv_block = fb - hkv * nbs
        # Compact: fb's low part is the compact index j; the q-range is read from sel_offsets
        # (passed as mKvToQOffsets) at [hkv, j] -- then j is remapped to the RAW slot via
        # sel_slots for page addressing / causal position downstream.
        base_blk = Int32(mKvToQOffsets[hkv, kv_block])
        count = Int32(mKvToQOffsets[hkv, kv_block + 1]) - base_blk
        q_lo = Int32(0)
        if is_first:
            q_lo = q_s
        q_hi = count
        if fb == fb_e:
            q_hi = q_e
        if q_hi > count:
            q_hi = count
        slice_count = q_hi - q_lo
        if slice_count < 0:
            slice_count = Int32(0)
        base = base_blk + q_lo
        total_rows = slice_count * self.qhead_per_kvhead
        n_tiles = (total_rows + self.m_block_size - 1) // self.m_block_size
        n_groups = (n_tiles + self.q_stage - 1) // self.q_stage
        # Compact: remap the compact index j -> raw slot (for page addressing + causal position).
        kv_block = Int32(self._mSelSlots[hkv, kv_block])
        return hkv, kv_block, base, total_rows, n_groups

    @cute.jit
    def _tile_rows(self, total_rows, g, stage):
        """The 128-row window for tile (g*q_stage + stage): rows [tt*128, +128), clamped.
        n_rows==0 marks an empty trailing stage of the last group (pipelines still run)."""
        tt = g * self.q_stage + stage
        row_start = tt * self.m_block_size
        n_rows = total_rows - row_start
        if n_rows > self.m_block_size:
            n_rows = Int32(self.m_block_size)
        if n_rows < 0:
            n_rows = Int32(0)
        return row_start, n_rows

    # ----------------------------------------------------------------- #
    # Packed gather/scatter: each tile row maps to (query_local, head_local) =
    # divmod(row_start+row, qhead). Q is token-major [Tq, Hq, D]; O/LSE partials
    # remain head-major [Hq, Tq, topK, ...] for the combine kernel.
    # ----------------------------------------------------------------- #
    @staticmethod
    @cute.jit
    def _q_stage_2d(sQ, stage):
        """Reshape one make_smem_layout_a sQ stage into a 2D [m_block, D] logical view.

        sQ shape: (MMA, MMA_Q, MMA_K, PIPE) where MMA=(M_ATOM, K_ATOM); the 2D view merges
        (K_ATOM, MMA_K) into the D mode so [qhead, K_ATOM] K-half boxes can be flat_divided
        out of it (see prototype/bench_tma_q_load.py for the layout derivation)."""
        s = sQ[None, None, None, stage]
        return cute.make_tensor(
            s.iterator,
            cute.make_layout(
                (s.shape[0][0], (s.shape[0][1], s.shape[2])),
                stride=(s.stride[0][0], (s.stride[0][1], s.stride[2])),
            ),
        )

    def _ring_depths(self, o_buffers):
        """Barrier-free smem ring depths, sized so a producer can never lap the slowest
        consumer (ordering rides the existing pipeline chains; only OVERWRITE needs depth).

        * pairs ring (sQIdxRank): produced by the load warp at Q-issue, consumed by softmax
          (post S-full) and the store warp (post sO-full). The loader leads the store warp
          by at most q_load_stage (Q ring, +1 for the pre-acquire gather) + 2*q_stage (s_p +
          o_acc rings through the mma) + o_buffers (sO ring) tiles.
        * stats ring (sStats): produced by softmax just BEFORE its P-release, consumed by
          the store warp post sO-full; softmax leads by at most 2*q_stage + o_buffers.
        Both get +2 margin."""
        pairs_depth = self.q_load_stage + 2 * self.q_stage + o_buffers + 3
        stats_depth = 2 * self.q_stage + o_buffers + 2
        return pairs_depth, stats_depth

    @cute.jit
    def _q_src_rowgroups(self, idx_ranks, hkv, base, row_start, n_rows, oob_rg, sPairs_slot):
        """Prefetch the tile's per-box source row groups: ONE scattered qidx load per 16-row
        box, src_rg = (qidx*Hq + hkv*qhead)/qhead (invalid tail boxes -> OOB, TMA zero-fill).

        Called BEFORE the Q slot's ``producer_acquire`` so the ~700-cycle gmem round trip of
        the qidx gather overlaps the acquire spin instead of sitting at the head of the
        post-acquire issue chain (it was the dominant term of the per-tile Q load cost).

        Lane 0 also caches each box's (qidx, rank) pair into ``sPairs_slot`` (the tile's
        sQIdxRank ring slot) so downstream consumers (softmax's (m~, l) export) read the
        indices from smem instead of re-gathering from gmem on their critical path. The
        rank load shares the qidx's cache line. Visibility: these plain smem stores are
        release-ordered by the Q barrier's arrive (expect_tx) and reach softmax through
        the Q-full -> QK -> S-full acquire chain."""
        qhead = const_expr(self.qhead_per_kvhead)
        nbox_m = const_expr(self.m_block_size // qhead)
        hq = Int32(idx_ranks.shape[0]) * qhead
        src_rgs = cute.make_rmem_tensor(nbox_m, Int32)
        lane0 = cute.arch.lane_idx() == 0
        for mb in cutlass.range_constexpr(nbox_m):
            src_rg = oob_rg
            if mb * qhead < n_rows:
                ql = (row_start + mb * qhead) // qhead
                qidx = Int32(idx_ranks[hkv, base + ql, 0])
                src_rg = (qidx * hq + hkv * qhead) // qhead
                if lane0:
                    sPairs_slot[mb, 0] = qidx
                    sPairs_slot[mb, 1] = Int32(idx_ranks[hkv, base + ql, 1])
            elif lane0:
                # OOB tail box: huge sentinel so the causal mask's min-qidx tile skip is
                # never vetoed by a stale slot value (OOB rows are store-dropped anyway).
                sPairs_slot[mb, 0] = Int32(0x3FFFFFFF)
                sPairs_slot[mb, 1] = Int32(0)
            src_rgs[mb] = src_rg
        return src_rgs

    @cute.jit
    def _gather_q_prefetch(self, mQ, idx_ranks, hkv, base, row_start, n_rows, tidx, gmem_tiled_copy, sPairs_slot):
        """Phase 1 of the cooperative cp.async Q gather (store-in-corr path): per-thread
        scattered qidx reads -> gmem row pointers, issued BEFORE the Q slot acquire so the
        gather latency overlaps the acquire spin. The box-leader thread of each 16-row
        group also writes the (qidx, rank) pair into the sQIdxRank ring (the causal mask
        consumes it post S-wait). Restored from the pre-store-warp 3-load-warp design."""
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        num_threads = gmem_tiled_copy.size
        qhead = const_expr(self.qhead_per_kvhead)
        # Pairs-ring write: one box per thread (tidx 0..nbox-1), independent of the
        # pointer-loop's row<->thread mapping (piggybacking on it mapped rows wrongly).
        if tidx < const_expr(self.m_block_size // qhead):
            box_row = tidx * qhead
            if box_row < n_rows:
                ql_b = (row_start + box_row) // qhead
                sPairs_slot[tidx, 0] = Int32(idx_ranks[hkv, base + ql_b, 0])
                sPairs_slot[tidx, 1] = Int32(idx_ranks[hkv, base + ql_b, 1])
            else:
                # OOB tail box: huge sentinel (never veto the causal row skip).
                sPairs_slot[tidx, 0] = Int32(0x3FFFFFFF)
                sPairs_slot[tidx, 1] = Int32(0)
        num_ptr = cute.ceil_div(cute.size(tQcQ_row), threads_per_row)
        tPrPtr = cute.make_rmem_tensor(num_ptr, Int64)
        for i in cutlass.range_constexpr(num_ptr):
            row = i * num_threads + tQcQ_row[tidx % threads_per_row][0]
            head = Int32(0)
            qidx = Int32(0)
            if row < n_rows:
                virtual = row_start + row
                ql = virtual // qhead
                hl = virtual % qhead
                qidx = Int32(idx_ranks[hkv, base + ql, 0])
                head = hkv * qhead + hl
            tPrPtr[i] = utils.elem_pointer(mQ, (qidx, head, 0)).toint()
        return tPrPtr

    @cute.jit
    def _gather_q_copy(self, mQ, sQ, stage, tPrPtr, n_rows, gmem_tiled_copy, tidx):
        """Phase 2: shuffle the row pointers across the row's thread set and issue the
        cp.async copies into the MMA-A swizzled sQ slot (pre-store-warp design)."""
        sQ_stage = sQ[None, None, None, stage]
        sQ_stage = cute.make_tensor(
            sQ_stage.iterator,
            cute.make_layout(
                (sQ_stage.shape[0][0], (sQ_stage.shape[0][1], sQ_stage.shape[2])),
                stride=(sQ_stage.stride[0][0], (sQ_stage.stride[0][1], sQ_stage.stride[2])),
            ),
        )
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQsQ = gmem_thr_copy.partition_D(sQ_stage)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_tiled_copy.get_slice(0).partition_S(cQ)
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            q_ptr_i64 = utils.shuffle_sync(tPrPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row)
            q_gmem_ptr = cute.make_ptr(mQ.element_type, q_ptr_i64, cute.AddressSpace.gmem, assumed_align=16)
            if t0QcQ[0, m, 0][0] < n_rows - tQcQ_row[0][0]:
                src = cute.make_tensor(q_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tQsQ.shape[0][0])
                src_copy = cute.tiled_divide(src, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                    ki = tQcQ[0, 0, k][1] // elems_per_load
                    cute.copy(gmem_thr_copy, src_copy[None, ki], tQsQ[None, m, k])

    @cute.jit
    def _q_write_pairs(self, idx_ranks, hkv, base, row_start, n_rows, tidx, sPairs_slot):
        """Box-leader (qidx, rank) write into the sQIdxRank ring (split out of
        _gather_q_prefetch; the causal mask consumes it post S-wait). One box per lane."""
        qhead = const_expr(self.qhead_per_kvhead)
        if tidx < const_expr(self.m_block_size // qhead):
            box_row = tidx * qhead
            if box_row < n_rows:
                ql_b = (row_start + box_row) // qhead
                sPairs_slot[tidx, 0] = Int32(idx_ranks[hkv, base + ql_b, 0])
                sPairs_slot[tidx, 1] = Int32(idx_ranks[hkv, base + ql_b, 1])
            else:
                sPairs_slot[tidx, 0] = Int32(0x3FFFFFFF)
                sPairs_slot[tidx, 1] = Int32(0)

    @cute.jit
    def _q_ptr_issue(self, idx_ranks, hkv, base, row_start, n_rows, tidx, gmem_tiled_copy):
        """Phase-1a: ISSUE the per-row scattered qidx gmem loads into a register fragment
        (no pointer math, no smem). Returns the in-flight qidx fragment; the consuming
        ``_q_ptr_resolve`` is deferred so the ~gmem latency overlaps the PREVIOUS tile's
        cp.async Q copy (a 1-tile software pipeline of the qidx gather; see sec 7al)."""
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        num_threads = gmem_tiled_copy.size
        qhead = const_expr(self.qhead_per_kvhead)
        num_ptr = cute.ceil_div(cute.size(tQcQ_row), threads_per_row)
        tQidx = cute.make_rmem_tensor(num_ptr, Int32)
        for i in cutlass.range_constexpr(num_ptr):
            row = i * num_threads + tQcQ_row[tidx % threads_per_row][0]
            qidx = Int32(0)
            if row < n_rows:
                ql = (row_start + row) // qhead
                qidx = Int32(idx_ranks[hkv, base + ql, 0])
            tQidx[i] = qidx
        return tQidx

    @cute.jit
    def _q_ptr_resolve(self, mQ, tQidx, hkv, row_start, n_rows, tidx, gmem_tiled_copy):
        """Phase-1b: CONSUME the in-flight qidx fragment -> per-row gmem row pointers
        (``elem_pointer`` reads tQidx, stalling only if the load issued by _q_ptr_issue is
        not yet back -- which it is, having overlapped the prior tile's copy). The head /
        row arithmetic is recomputed here (no gmem dependency)."""
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        num_threads = gmem_tiled_copy.size
        qhead = const_expr(self.qhead_per_kvhead)
        num_ptr = cute.ceil_div(cute.size(tQcQ_row), threads_per_row)
        tPrPtr = cute.make_rmem_tensor(num_ptr, Int64)
        for i in cutlass.range_constexpr(num_ptr):
            row = i * num_threads + tQcQ_row[tidx % threads_per_row][0]
            head = Int32(0)
            qidx = Int32(0)
            if row < n_rows:
                hl = (row_start + row) % qhead
                head = hkv * qhead + hl
                qidx = tQidx[i]
            tPrPtr[i] = utils.elem_pointer(mQ, (qidx, head, 0)).toint()
        return tPrPtr

    @cute.jit
    def _load_tma_Q_packed(self, sQ, stage, gQ_box, tma_atom_Q, bar, src_rgs):
        """TMA-load one 128-row packed Q tile as nbox_m x n_kblk [qhead, K_ATOM] K-half boxes
        (serial issue; G2S TMA copies elect ONE lane per warp, so per-lane parallel issue
        silently drops boxes -- tile-level parallelism comes from splitting tiles across the
        two load warps instead, see load()).

        Box mb covers packed rows [mb*qhead, +qhead) == ONE query's full qhead-group, which is
        contiguous in the flat [Tq*Hq, D] Q view at the prefetched row group ``src_rgs[mb]``
        (see _q_src_rowgroups). All copies land on the same stage barrier (tx adds atomically);
        OOB row groups are zero-filled with the bytes still delivered, keeping the tx-count
        uniform."""
        qhead = const_expr(self.qhead_per_kvhead)
        k_atom = const_expr(1024 // self.q_dtype.width)
        nbox_m = const_expr(self.m_block_size // qhead)
        n_kblk = const_expr(self.head_dim_padded // k_atom)
        sQ_box = cute.flat_divide(self._q_stage_2d(sQ, stage), (qhead, k_atom))
        bQS, bQG = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_box, 0, 2),
            cute.group_modes(gQ_box, 0, 2),
        )
        for mb in cutlass.range_constexpr(nbox_m):
            for kb in cutlass.range_constexpr(n_kblk):
                cute.copy(
                    tma_atom_Q,
                    bQG[None, src_rgs[mb], kb],
                    bQS[None, mb, kb],
                    tma_bar_ptr=bar,
                )

    @cute.jit
    def _store_O(self, tma_atom_O, bSG_sO, bSG_gO, hkv, base, row_start, n_rows, box_idx, tqk):
        """Store the o_nbox swizzled [qhead, D] sO boxes to O_partial via [qhead, D] bulk-TMA
        (make_tiled_tma_atom idiom), one box per issuing thread (box_idx 0..o_nbox-1; callers
        may pass a shifted/negative index for non-issuing lanes).

        TILE-ORDERED layout (sec 7t): O_partial is flat [Hkv * Tq*topK * qhead, D] indexed by
        pair position p = base + ql -- box b's dest row group is simply hkv*tqk + base + ql
        (tqk = Tq*topK), a pure address computation with NO scattered qidx/rank gmem reads
        and no division. The combine resolves (q, rank) -> p through the inverse map.

        Every issuing lane issues + commits EXACTLY ONE box per tile; an invalid box (its
        rows >= n_rows) is sent to a one-past-end row group so the TMA drops it (OOB-drop, no
        write). Uniform per-lane commit counts keep the deferred wait correct across
        partial/empty tiles and block boundaries. The matching wait is NOT here: the issuer
        drains its own group before the buffer's next reuse."""
        box = const_expr(self.qhead_per_kvhead)
        nbox = const_expr(self.o_nbox)
        if box_idx >= 0 and box_idx < nbox:
            dest = Int32(const_expr(self.nheads_kv)) * tqk  # one-past-end -> TMA drops the box
            if box_idx * box < n_rows:
                ql = (row_start + box_idx * box) // box
                dest = hkv * tqk + base + ql
            # NOTE: an EVICT_FIRST cache hint here measured neutral-to-negative (sec 7ab):
            # reads are L2-resident at these shapes (DRAM reads ~18 MB), so the store's cost
            # is write-stream QUEUE CONTENTION, which no cache policy removes.
            cute.copy(tma_atom_O, bSG_sO[None, box_idx], bSG_gO[None, dest])
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _tma_evict_first_policy(self):
        return Int64(const(int(cute.CacheEvictionPriority.EVICT_FIRST)))

    @cute.jit
    def _load_tma_evict_first(self, tma_atom, tXs, tXg, pipeline_kv, producer_state, page):
        stage = producer_state.index
        pipeline_kv.producer_acquire(producer_state)
        bar = pipeline_kv.producer_get_barrier(producer_state)
        cute.copy(
            tma_atom,
            tXg[None, 0, page],
            tXs[None, stage],
            tma_bar_ptr=bar,
            cache_policy=self._tma_evict_first_policy(),
        )

    @cute.jit
    def _load_block_paged(
        self,
        tma_atom,
        tXs,
        tXg,
        pipeline_kv,
        producer_state,
        hkv,
        kv_block,
        mTopkSlotIds,
        denom,
    ):
        """Fill ONE 128-key smem stage from ``ratio`` page-sized TMAs (page_size < 128).

        ``tXs`` is the smem tensor pre-partitioned to ``(box, page, stage)`` and ``tXg`` to
        ``(box, num_pages)`` (done ONCE in load(), not per page — hd512's load_inner_paged_vt
        pattern). One ``producer_acquire`` (the stage barrier already expects the full 128-key
        tx_count); the ``ratio`` page-box copies' byte-deliveries sum to it. The smem box is a
        swizzle-preserving page stripe of the live layout, so the writes land exactly where the
        128-key MMA reads."""
        stage = producer_state.index
        pipeline_kv.producer_acquire(producer_state)
        bar = pipeline_kv.producer_get_barrier(producer_state)
        for sub in cutlass.range_constexpr(self.ratio):
            page = Int32(mTopkSlotIds[hkv, kv_block * self.ratio + sub]) // denom
            cute.copy(
                tma_atom,
                tXg[None, page],
                tXs[None, sub, stage],
                tma_bar_ptr=bar,
                cache_policy=self._tma_evict_first_policy(),
            )

    # ----------------------------------------------------------------- #
    # Load: KV-stationary, load-balanced. Each work-item is a run of blocks (from the
    # scheduler's start/end); for each block load its 128-key K/V ONCE, then stream the
    # block-slice's 128-row Q-tiles (q_stage at a time for ping-pong).
    # ----------------------------------------------------------------- #
    @cute.jit
    def load(
        self,
        mQ,
        mK,
        mV,
        sQ,
        sK,
        sV,
        tma_atom_K,
        tma_atom_V,
        tma_atom_Q,
        mQ_tma,
        pipeline_q,
        pipeline_kv,
        thr_mma_qk,
        thr_mma_pv,
        mTopkSlotIds,
        mKvToQOffsets,
        mKvToQIdxRank,
        mWorkStart,
        mWorkEnd,
        nbs,
        mSeqUsedK,
        n_batches,
        sQIdxRank,
        tile_scheduler,
        gmem_tiled_copy_Q=None,
    ):
        tidx = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE
        # Cooperative-gather thread id over the 3 load warps (96 contiguous threads; the
        # mod maps them bijectively onto [0, 96)).
        tidx96 = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * (len(self.load_warp_ids) + 1))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        issue_kv = warp_idx == self.load_warp_ids[0]
        # Free-running Q-tile ordinal (across blocks / work items); both load warps advance
        # it uniformly. slot = ctr % pairs_depth indexes the sQIdxRank ring. Softmax keeps an
        # identically-ordered counter, so producer and consumers agree on slots.
        pairs_ctr = Int32(0)
        # Q-tile ownership by stage parity: warp 13 issues stage-0 tiles (+ K/V), warp 14
        # stage-1 tiles. Both warps advance the producer state uniformly; only the owner
        # acquires (arms expect-tx) and issues, so each slot has exactly one producer.
        q_warp_for_stage = (self.load_warp_ids[0], self.load_warp_ids[-1])
        denom = self.nheads_kv * self.page_size
        q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.q_load_stage)
        kv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        # Q gmem [Tq*Hq, D] tiled into [qhead, K_ATOM] K-half boxes:
        # ((qhead, K_ATOM), n_row_groups, n_kblk). OOB row group -> TMA zero-fill.
        k_atom = const_expr(1024 // self.q_dtype.width)
        gQ_box = cute.local_tile(mQ_tma, (const_expr(self.qhead_per_kvhead), k_atom), (None, None))
        oob_rg = Int32(cute.size(gQ_box, mode=[1]))
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            wi, _h, _b, _s = work_tile.tile_idx
            valid, fb_s, fb_e, q_s, q_e, num_fb = self._decode_workitem(wi, mWorkStart, mWorkEnd, nbs)
            num_real = Int32(0)
            if valid:
                num_real = self._num_real_cj(fb_s, fb_e, nbs)
            fb = fb_s
            for i in cutlass.range(num_real, unroll=1):
                if i > 0:  # advance to next REAL block, skipping the msb-padding gap
                    fb = self._next_cj(fb, nbs)
                hkv, kv_block, base, total_rows, n_groups = self._slice_block(
                    fb, i == 0, fb_e, q_s, q_e, nbs, mKvToQOffsets
                )
                if n_groups > 0:  # skip empty (unselected) block-slots / empty slices
                    # Per-block KV addressing (paged TMA).
                    mK_cur, mV_cur = [t[None, None, hkv, None] for t in (mK, mV)]
                    page = Int32(0)
                    if const_expr(self.ratio == 1):
                        # page_size == 128: a single TMA fills the whole 128-key block buffer.
                        gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None))
                        gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None))
                        tSgK = thr_mma_qk.partition_B(gK)
                        tOgV = thr_mma_pv.partition_B(gV)
                        tKsK, tKgK = cpasync.tma_partition(
                            tma_atom_K,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(tSgK, 0, 3),
                        )
                        tVsV, tVgV = cpasync.tma_partition(
                            tma_atom_V,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sV, 0, 3),
                            cute.group_modes(tOgV, 0, 3),
                        )
                        page = Int32(mTopkSlotIds[hkv, kv_block * self.ratio]) // denom
                    else:
                        # page_size < 128: ratio pages per 128-key block. Page-box gmem (page
                        # selected via mTopkSlotIds) + the live 128-key smem flat_divided so the
                        # page becomes a separate mode (_load_block_paged slices one page each).
                        # K's gmem uses a page-N tiled_mma matching the page-N atom built in
                        # __call__; V uses the full PV mma (paged on the contraction; V-map over
                        # head_dim_v is unaffected, the hd512 Vt pattern).
                        ps = self.page_size
                        mma_tiler_qk_pg = (self.mma_tiler_qk[0], ps, self.mma_tiler_qk[2])
                        mma_tiler_pv_pg = (self.mma_tiler_pv[0], self.mma_tiler_pv[1], ps)
                        tiled_mma_qk_pg = sm100_utils_basic.make_trivial_tiled_mma(
                            self.q_dtype,
                            tcgen05.OperandMajorMode.K,
                            tcgen05.OperandMajorMode.K,
                            self.qk_acc_dtype,
                            tcgen05.CtaGroup.ONE,
                            mma_tiler_qk_pg[:2],
                        )
                        thr_qk_pg = tiled_mma_qk_pg.get_slice(0)
                        gK = cute.flat_divide(mK_cur, (mma_tiler_qk_pg[1], mma_tiler_qk_pg[2]))[None, None, 0, 0, None]
                        gV = cute.flat_divide(mV_cur, (mma_tiler_pv_pg[1], mma_tiler_pv_pg[2]))[None, None, 0, 0, None]
                        tSgK = thr_qk_pg.partition_B(gK)
                        tOgV = thr_mma_pv.partition_B(gV)
                        k_box = tiled_mma_qk_pg.partition_shape_B(cute.dice(mma_tiler_qk_pg, (None, 1, 1)))
                        fdK = cute.flat_divide(sK, k_box)
                        v_per = cute.size(sV, mode=[2]) // self.ratio
                        fdV = cute.flat_divide(sV, (cute.size(sV, mode=[0]), 1, v_per))
                        # (box, page, stage) smem views (page = the size-ratio mode popped by
                        # flat_divide: K mode 3, V mode 5). Partition ONCE per K/V (not per page).
                        viewK = fdK[None, None, None, (None, 0), 0, 0, None]
                        viewV = fdV[None, None, None, 0, 0, None, None]
                        tKsK, tKgK = cpasync.tma_partition(
                            tma_atom_K,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(viewK, 0, 3),
                            cute.group_modes(tSgK, 0, 3),
                        )
                        tVsV, tVgV = cpasync.tma_partition(
                            tma_atom_V,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(viewV, 0, 3),
                            cute.group_modes(tOgV, 0, 3),
                        )
                    # KV-stationary load order: K, then the FIRST Q tile, then V, then the rest
                    # (K, Q0, V, Q1.., per block). Streaming Q0 before V lets the MMA issue this
                    # block's first QK and run its deferred PV -- which releases the PREVIOUS block's
                    # V (same kv buffer 1) -- before the loader's V acquire needs it, breaking the
                    # load<->mma cycle. V still lands right after Q0, overlapping the remaining Q
                    # gathers so the later PVs don't stall.
                    if issue_kv:
                        if const_expr(self.ratio == 1):
                            self._load_tma_evict_first(tma_atom_K, tKsK, tKgK, pipeline_kv, kv_producer_state, page)
                        else:
                            self._load_block_paged(
                                tma_atom_K,
                                tKsK,
                                tKgK,
                                pipeline_kv,
                                kv_producer_state,
                                hkv,
                                kv_block,
                                mTopkSlotIds,
                                denom,
                            )
                        kv_producer_state.advance()
                    # Stream the block-slice's Q-tiles, q_stage (ping-pong) at a time, split
                    # across the two load warps by stage parity (issue-throughput; see above).
                    # The tile's qidx gather (scattered gmem loads) is issued BEFORE the slot
                    # acquire so its latency overlaps the acquire spin (see _q_src_rowgroups).
                    for g in cutlass.range(n_groups, unroll=1):
                        # Cooperative cp.async gather (all 3 load warps, 96 threads),
                        # Pipeline qidx across the group's two tiles: issue
                        # BOTH tiles' scattered qidx gmem loads up front, then
                        # resolve+copy tile 0 while tile 1's qidx loads are in flight,
                        # so tile 1's gather latency overlaps tile 0's cp.async copy
                        # (vs the old "gather i -> copy i" which exposed the gather
                        # whenever the slot acquire didn't spin). _q_write_pairs feeds
                        # the causal mask; _q_ptr_issue/_resolve split the per-row
                        # pointer gather into issue (loads) + consume (elem_pointer).
                        rs0, nr0 = self._tile_rows(total_rows, g, Int32(0))
                        rs1, nr1 = self._tile_rows(total_rows, g, Int32(1))
                        ps0 = pairs_ctr % const_expr(self.pairs_depth)
                        ps1 = (pairs_ctr + 1) % const_expr(self.pairs_depth)
                        pairs_ctr += 2
                        tQ0 = self._q_ptr_issue(mKvToQIdxRank, hkv, base, rs0, nr0, tidx96, gmem_tiled_copy_Q)
                        tQ1 = self._q_ptr_issue(mKvToQIdxRank, hkv, base, rs1, nr1, tidx96, gmem_tiled_copy_Q)
                        self._q_write_pairs(mKvToQIdxRank, hkv, base, rs0, nr0, tidx96, sQIdxRank[ps0, None, None])
                        self._q_write_pairs(mKvToQIdxRank, hkv, base, rs1, nr1, tidx96, sQIdxRank[ps1, None, None])
                        # --- tile 0 ---
                        qstage = q_producer_state.index
                        p0 = self._q_ptr_resolve(mQ, tQ0, hkv, rs0, nr0, tidx96, gmem_tiled_copy_Q)
                        pipeline_q.producer_acquire_w_index_phase(qstage, q_producer_state.phase)
                        self._gather_q_copy(mQ, sQ, qstage, p0, nr0, gmem_tiled_copy_Q, tidx96)
                        cute.arch.cp_async_commit_group()
                        pipeline_q.sync_object_full.arrive_cp_async_mbarrier(qstage)
                        q_producer_state.advance()
                        # This block's V right after its first Q tile (ordering note above).
                        if g == 0 and issue_kv:
                            if const_expr(self.ratio == 1):
                                self._load_tma_evict_first(
                                    tma_atom_V, tVsV, tVgV, pipeline_kv, kv_producer_state, page
                                )
                            else:
                                self._load_block_paged(
                                    tma_atom_V,
                                    tVsV,
                                    tVgV,
                                    pipeline_kv,
                                    kv_producer_state,
                                    hkv,
                                    kv_block,
                                    mTopkSlotIds,
                                    denom,
                                )
                            kv_producer_state.advance()
                        # --- tile 1 (its qidx latency overlapped tile 0's copy) ---
                        qstage = q_producer_state.index
                        p1 = self._q_ptr_resolve(mQ, tQ1, hkv, rs1, nr1, tidx96, gmem_tiled_copy_Q)
                        pipeline_q.producer_acquire_w_index_phase(qstage, q_producer_state.phase)
                        self._gather_q_copy(mQ, sQ, qstage, p1, nr1, gmem_tiled_copy_Q, tidx96)
                        cute.arch.cp_async_commit_group()
                        pipeline_q.sync_object_full.arrive_cp_async_mbarrier(qstage)
                        q_producer_state.advance()
            work_tile = tile_scheduler.advance_to_next_work()
        if issue_kv:
            pipeline_kv.producer_tail(kv_producer_state)
            # Q tail from ONE warp only: producer_tail arms expect-tx internally, so a second
            # warp's tail would double-arrive the stage barriers (hardware mbarrier error).
            # Both warps advanced q_producer_state uniformly, so warp 13's state is global.
            pipeline_q.producer_tail(q_producer_state)

    # ----------------------------------------------------------------- #
    # MMA: one-ahead software pipeline over the flat (block, group) tile stream. Each
    # iteration issues this group's QK *after* the previous group's PV, so the previous
    # group's softmax latency -- and, at block boundaries, the next block's QK -- overlaps
    # the current PV. The two q_stage tmem buffers are reused every group (they hold the
    # in-flight group), like FA4's mainloop; we just defer the PV by one group and swap K/V
    # at block boundaries. K and V occupy fixed kv-pipeline slots (K -> idx 0, V -> idx 1;
    # sV aliases sK), so they are tracked as two independent single-buffer sub-streams: the
    # next block's K can be waited while the current block's V is still resident (the
    # cross-block overlap), which kv_stage=2 holds since K[b+1] and V[b] are different slots.
    # ----------------------------------------------------------------- #
    @cute.jit
    def mma(
        self,
        tiled_mma_qk,
        tiled_mma_pv,
        sQ,
        sK,
        sV,
        tStS,
        tOtO,
        tOrP,
        pipeline_q,
        pipeline_kv,
        pipeline_s_p,
        pipeline_o_acc,
        mKvToQOffsets,
        mWorkStart,
        mWorkEnd,
        nbs,
        mSeqUsedK,
        n_batches,
        tile_scheduler,
    ):
        import flash_attn.cute.blackwell_helpers as sm100_utils

        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        # Debug: per-CTA / per-thread identity for the cute.printf logs below.
        # dbg_bx, dbg_by, dbg_bz = cute.arch.block_idx()
        # dbg_tx = cute.arch.thread_idx()[0]

        mma_q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_load_stage)
        # K -> kv-buffer 0, V -> kv-buffer 1 (sV aliases sK); tracked as two independent
        # single-buffer sub-streams with their own consumer phase (toggled per block).
        KBUF = const_expr(0)
        VBUF = const_expr(1)

        # S/P/O tmem-stage + pipeline index/phase tracked via per-role pipeline states (replacing
        # the hand-rolled so_stage / po_phase):
        #   qk_state (Producer): QK target tmem stage + s_p S-full commit; advances once per QK.
        #   pv_state (Producer): PV target tmem stage + o_acc O buffer. The mma owns the O tmem
        #       buffer: producer_acquire (wait correction done reading the prior O = WAR hazard,
        #       the former "O rescaled" / s_o signal) BEFORE the PV, then producer_commit (O full)
        #       AFTER it. Advances once per PV. The o_acc empty barrier is producer-pre-armed at
        #       init, so the Producer phase (starting 1) is correct here -- no consumer pre-release.
        #   sp_state (Consumer): s_p P-full acquire (index + phase); advances once per PV. Consumer-
        #       init (phase 0): s_p's empty is armed by softmax's first real P, not a pre-init.
        # qk_state runs one tile ahead of the pv/sp pair (the one-ahead deferral). Producer states
        # use .index for commit; pv_state also uses .phase for its acquire.
        qk_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.q_stage)
        pv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.q_stage)
        sp_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_stage)
        k_phase = Int32(0)
        v_phase = Int32(0)

        is_prologue = Int32(1)
        has_epilogue = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            wi, _h, _b, _s = work_tile.tile_idx
            # cute.printf("[b=(%d,%d,%d) t=%d] new work tile: wi: %d, _h: %d, _b: %d, _s: %d\n",
            #             dbg_bx, dbg_by, dbg_bz, dbg_tx, wi, _h, _b, _s)
            valid, fb_s, fb_e, q_s, q_e, num_fb = self._decode_workitem(wi, mWorkStart, mWorkEnd, nbs)
            num_real = Int32(0)
            if valid:
                num_real = self._num_real_cj(fb_s, fb_e, nbs)
            fb = fb_s
            for i in cutlass.range(num_real, unroll=1):
                if i > 0:  # advance to next REAL block, skipping the msb-padding gap
                    fb = self._next_cj(fb, nbs)
                hkv, kv_block, base, total_rows, n_groups = self._slice_block(
                    fb, i == 0, fb_e, q_s, q_e, nbs, mKvToQOffsets
                )
                if n_groups > 0:
                    has_epilogue = Int32(1)
                    pipeline_kv.consumer_wait_w_index_phase(Int32(KBUF), k_phase)
                    k_phase ^= 1

                    # Prologue, S1 = Q1@K
                    if is_prologue:
                        qstage = mma_q_consumer_state.index
                        pipeline_q.consumer_wait_w_index_phase(qstage, mma_q_consumer_state.phase)
                        sm100_utils.gemm(
                            tiled_mma_qk,
                            tStS[None, None, None, qk_state.index],
                            tSrQ[None, None, None, qstage],
                            tSrK[None, None, None, KBUF],
                            zero_init=True,
                        )
                        pipeline_s_p.producer_commit_w_index(qk_state.index)
                        pipeline_q.consumer_release_w_index(qstage)
                        mma_q_consumer_state.advance()
                        qk_state.advance()

                    # Mainloop
                    for g in cutlass.range(n_groups * 2 - is_prologue, unroll=1):
                        # Si+1 = Qi+1 @ K
                        qstage = mma_q_consumer_state.index
                        pipeline_q.consumer_wait_w_index_phase(qstage, mma_q_consumer_state.phase)
                        sm100_utils.gemm(
                            tiled_mma_qk,
                            tStS[None, None, None, qk_state.index],
                            tSrQ[None, None, None, qstage],
                            tSrK[None, None, None, KBUF],
                            zero_init=True,
                        )
                        pipeline_s_p.producer_commit_w_index(qk_state.index)
                        pipeline_q.consumer_release_w_index(qstage)
                        if g == n_groups * 2 - is_prologue - 1:
                            # Release this block's K after its last QK.
                            pipeline_kv.consumer_release_w_index(Int32(KBUF))
                        mma_q_consumer_state.advance()
                        qk_state.advance()

                        # --- PV of the previous (deferred) group, overlapping this QK ---
                        if g == 1 - is_prologue:
                            pipeline_kv.consumer_wait_w_index_phase(Int32(VBUF), v_phase)
                            v_phase ^= 1

                        # Oi = Pi @ V. Consume P-full (s_p, from softmax) and acquire the O tmem
                        # buffer (o_acc empty = correction done reading the prior O); then produce O
                        # and commit O-full (o_acc) for correction.
                        pipeline_s_p.producer_acquire_w_index_phase(sp_state.index, sp_state.phase)
                        pipeline_o_acc.producer_acquire_w_index_phase(pv_state.index, pv_state.phase)
                        sm100_utils.gemm(
                            tiled_mma_pv,
                            tOtO[None, None, None, pv_state.index],
                            tOrP[None, None, None, pv_state.index],
                            tOrV[None, None, None, VBUF],
                            zero_init=True,
                        )
                        pipeline_o_acc.producer_commit_w_index(pv_state.index)
                        if not is_prologue and g == 0:
                            pipeline_kv.consumer_release_w_index(Int32(VBUF))
                        sp_state.advance()
                        pv_state.advance()

                    # Prologue has been processed.
                    is_prologue = Int32(0)

            # Move to the next tile.
            work_tile = tile_scheduler.advance_to_next_work()

        # Epilogue: flush the final pending group's PV (last tile's last group).
        if has_epilogue:
            pipeline_s_p.producer_acquire_w_index_phase(sp_state.index, sp_state.phase)
            pipeline_o_acc.producer_acquire_w_index_phase(pv_state.index, pv_state.phase)
            sm100_utils.gemm(
                tiled_mma_pv,
                tOtO[None, None, None, pv_state.index],
                tOrP[None, None, None, pv_state.index],
                tOrV[None, None, None, VBUF],
                zero_init=True,
            )
            pipeline_o_acc.producer_commit_w_index(pv_state.index)
            pipeline_kv.consumer_release_w_index(Int32(VBUF))

    @cute.jit
    def _softmax_step(
        self,
        mma_si_consumer_phase,
        n_block,
        softmax,
        thr_mma_qk,
        pipeline_s_p,
        thr_tmem_load,
        thr_tmem_store,
        tStS_t2r,
        tStP_r2t,
        stage,
        mask_fn=None,
        is_first=True,
    ):
        """Single-phase softmax step (forked from FA4's softmax_step, sm_stats signaling removed).

        Waits S (s_p full = QK done), masks, computes row_max, converts to P (exp2) into tmem,
        releases s_p (P ready for the PV), updates row_sum. The row_max/row_sum -> sScale handoff
        and its sync now live in softmax_loop / correction as a single producer_commit /
        consumer_wait on pipeline_sm_stats; this no longer touches sm_stats_barrier (the two-phase
        rescale signal is unused for the single-128-key-block design). softmax.row_max / row_sum
        are written to sScale by the loop."""
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScP_shape = (cta_qk_tiler[0], tilePlikeFP32)
        pipeline_s_p.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        pipeline_s_p.consumer_release_w_index(stage)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return mma_si_consumer_phase ^ 1

    # ----------------------------------------------------------------- #
    # Softmax: inner Q-tile loop; ratio sub-blocks per Q-tile (online).
    # ----------------------------------------------------------------- #
    @cute.jit
    def softmax_loop(
        self,
        stage,
        softmax_scale_log2,
        softmax_scale,
        thr_mma_qk,
        tStS,
        mM,
        mL,
        sQIdxRank,
        pipeline_s_p,
        block_info,
        SeqlenInfoCls,
        AttentionMaskCls,
        aux_tensors,
        mKvToQOffsets,
        mKvToQIdxRank,
        mCuSeqlensQ,
        mSeqUsedK,
        n_batches,
        mWorkStart,
        mWorkEnd,
        nbs,
        total_q,
        tile_scheduler,
    ):
        # Each softmax warpgroup is dispatched with its stage (0 or 1); per block it walks
        # the block-slice's Q-tile list, handling tile (g*q_stage + stage) of each group.
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        tSAcc = tStS[(None, None), 0, 0, stage]
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype)
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tmem_store_scale_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32)
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(tidx)
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        mma_si_consumer_phase = Int32(0)
        qhead = const_expr(self.qhead_per_kvhead)
        # Q-tile ordinal mirroring load()'s pairs_ctr (identical traversal): this
        # warpgroup's tile for group g is ordinal (ctr + stage); ctr += q_stage per group.
        pairs_ctr = Int32(0)

        seqlen = SeqlenInfoCls(Int32(0))
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            wi, _h, _b, _s = work_tile.tile_idx
            valid, fb_s, fb_e, q_s, q_e, num_fb = self._decode_workitem(wi, mWorkStart, mWorkEnd, nbs)
            num_real = Int32(0)
            if valid:
                num_real = self._num_real_cj(fb_s, fb_e, nbs)
            fb = fb_s
            for blk in cutlass.range(num_real, unroll=1):
                if blk > 0:  # advance to next REAL block, skipping the msb-padding gap
                    fb = self._next_cj(fb, nbs)
                hkv, kv_block, base, total_rows, n_groups = self._slice_block(
                    fb, blk == 0, fb_e, q_s, q_e, nbs, mKvToQOffsets
                )
                if n_groups > 0:
                    # Per-block column limit (same for all its Q-tiles), using the REAL per-seq
                    # KV length Lk_b = mSeqUsedK[b] (not a uniform msb*128). batch b = kv_block //
                    # msb (msb = nbs // n_batches, uniform padded block-slots per seq); within-seq
                    # block sb = kv_block - b*msb (absolute key position sb*128 + col).
                    #   causal:     col <= qidx + col_limit_base, col_limit_base = (Lk_b - Tq_b)
                    #               - q_off - sb*128 (query pos = (t-q_off)+(Lk_b-Tq_b), suffix).
                    #   non-causal: col <= col_limit_base = (Lk_b - 1) - sb*128 (key pos < Lk_b;
                    #               drops the partial last block's padding columns).
                    msb = nbs // n_batches
                    b = kv_block // msb
                    sb = kv_block - b * msb
                    Lkb = Int64(mSeqUsedK[b])
                    if const_expr(self.sparse_causal):
                        q_off = Int64(mCuSeqlensQ[b])
                        tq_b = Int64(mCuSeqlensQ[b + 1]) - q_off
                        col_limit_base = (Lkb - tq_b) - q_off - sb * self.block_size
                    else:
                        col_limit_base = (Lkb - Int64(1)) - sb * self.block_size

                    for g in cutlass.range(n_groups, unroll=1):
                        row_start, n_rows = self._tile_rows(total_rows, g, stage)
                        # This tile's slot in the load warp's qidx/rank smem ring (valid to
                        # read only after the S-full wait inside the step; see _apply_mask).
                        pairs_slot = (pairs_ctr + stage) % const_expr(self.pairs_depth)
                        pairs_ctr += const_expr(self.q_stage)
                        # Always mask: causal uses the per-query suffix limit; non-causal applies
                        # the Lk_b key-padding limit (a no-op on full, non-last blocks).
                        mask_fn = partial(
                            self._apply_mask,
                            thr_mma_qk=thr_mma_qk,
                            thr_tmem_load=thr_tmem_load,
                            row_start=row_start,
                            n_rows=n_rows,
                            sPairs_slot=sQIdxRank[pairs_slot, None, None],
                            col_limit_base=col_limit_base,
                        )
                        softmax = SoftmaxSm100.create(softmax_scale_log2, rescale_threshold=8.0)
                        softmax.reset()
                        mma_si_consumer_phase = self._softmax_step(
                            mma_si_consumer_phase,
                            Int32(0),
                            softmax,
                            thr_mma_qk,
                            pipeline_s_p,
                            thr_tmem_load,
                            thr_tmem_store,
                            tStS_t2r,
                            tStP_r2t,
                            stage,
                            mask_fn=mask_fn,
                            is_first=True,
                        )
                        # Export this tile's per-row (m~, l) DIRECTLY to gmem (deferred
                        # normalization: the combine consumes them; correction has no softmax
                        # dependency at all). TILE-ORDERED layout (sec 7t): the destination is
                        # segment-linear ((hkv, base*qhead + virtual_row) in the flat
                        # [Hkv, Tq*topK*qhead] stats planes), so the 128 threads issue two
                        # perfectly COALESCED 4B stores -- no qidx/rank gather, no division.
                        row_sum = softmax.row_sum[0]
                        row_max = softmax.row_max[0]
                        bad = row_sum == 0.0 or row_sum != row_sum
                        m_tilde = (row_max * softmax_scale_log2) if not bad else -Float32.inf
                        l_val = row_sum if not bad else Float32(0.0)
                        if tidx < n_rows:
                            # elem_pointer + 1-elem tensor (not subscript-assign): the DSL's
                            # dynamic-region closure rewrite loses the tensor param otherwise.
                            seg = base * qhead + row_start + tidx
                            m_ptr = utils.elem_pointer(mM, (hkv, seg)).toint()
                            l_ptr = utils.elem_pointer(mL, (hkv, seg)).toint()
                            m_gmem = cute.make_ptr(Float32, m_ptr, cute.AddressSpace.gmem, assumed_align=4)
                            l_gmem = cute.make_ptr(Float32, l_ptr, cute.AddressSpace.gmem, assumed_align=4)
                            cute.make_tensor(m_gmem, (1,))[0] = m_tilde
                            cute.make_tensor(l_gmem, (1,))[0] = l_val
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage,
        m_block,
        seqlen_q,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur=None,
        gO=None,
        gmem_tiled_copy_O=None,
    ):
        """Local copy of FA4's correction_epilogue (evac path only): load the O accumulator from
        tmem, multiply by ``scale`` (= 1/row_sum), cast to o_dtype, and store to the ``sO`` smem
        buffer for the subsequent bulk-TMA scatter. The gmem-store branch (mO_cur/gO/...) is unused
        by this kernel (it does its own _store_O scatter), so it is omitted. Brought in-kernel so it
        can be refactored independently of the shared FA4 method."""
        corr_tile_size = 8 * 32 // self.o_dtype.width
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(self.head_dim_v_padded // corr_tile_size, unroll_full=True):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    # ----------------------------------------------------------------- #
    # Correction: inner Q-tile loop; evacuate the RAW O accumulator to sO.
    # (O normalization is deferred to the combine, which consumes the (m~, l)
    # stats exported by softmax -- correction has NO softmax dependency.)
    # ----------------------------------------------------------------- #
    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk,
        thr_mma_pv,
        tStS,
        tOtO,
        sO,
        pipeline_o_acc,
        pipeline_sO,
        mKvToQOffsets,
        mWorkStart,
        mWorkEnd,
        nbs,
        mSeqUsedK,
        n_batches,
        tile_scheduler,
        tma_atom_O=None,
        mO_tma=None,
        tqk=None,
    ):
        # Per sO buffer: a merged [m_block, D] swizzled view over its o_nbox x [qhead, D]
        # boxes so the stock correction_epilogue evac fills it unchanged. Tiles rotate
        # through the o_buffers buffers in order (Producer pipeline state); the buffer's
        # evac view is built per tile from the DYNAMIC buffer index (layouts are identical
        # across buffers).
        # Store-in-correction: correction stores O itself. The sO ring needs no mbarriers (the
        # producer (evac) and consumer (TMA read) are this warpgroup); buffer reuse is ordered
        # by a per-thread rolling wait_group(o_buffers-1) on the issuing threads + a WG named
        # barrier. Box b is issued by thread b (tidx 0..7).
        sO_evac_layout = cute.group_modes(cute.select(sO[None, None, None, 0].layout, mode=[0, 2, 1]), 0, 2)
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        o_phase = Int32(0)
        so_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.o_buffers)
        gO_bulk_c = cute.local_tile(mO_tma, (const_expr(self.qhead_per_kvhead), self.head_dim_v_padded), (None, 0))
        gO_grp_c = cute.group_modes(gO_bulk_c, 0, 2)
        corr_bar = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.Epilogue),
            num_threads=cute.arch.WARP_SIZE * len(self.correction_warp_ids),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            wi, _h, _b, _s = work_tile.tile_idx
            valid, fb_s, fb_e, q_s, q_e, num_fb = self._decode_workitem(wi, mWorkStart, mWorkEnd, nbs)
            num_real = Int32(0)
            if valid:
                num_real = self._num_real_cj(fb_s, fb_e, nbs)
            fb = fb_s
            for blk in cutlass.range(num_real, unroll=1):
                if blk > 0:  # advance to next REAL block, skipping the msb-padding gap
                    fb = self._next_cj(fb, nbs)
                hkv, kv_block, base, total_rows, n_groups = self._slice_block(
                    fb, blk == 0, fb_e, q_s, q_e, nbs, mKvToQOffsets
                )
                if n_groups > 0:
                    for g in cutlass.range(n_groups, unroll=1):
                        # Per group, handle q_stage tiles (one per softmax warpgroup). sO is
                        # multi-buffered (o_buffers independent of q_stage); tiles rotate
                        # through the buffers in order via so_state.
                        for stage in cutlass.range_constexpr(self.q_stage):
                            buf = so_state.index
                            row_start, n_rows = self._tile_rows(total_rows, g, stage)
                            # Deferred normalization: the combine divides by l, so the evac is a
                            # RAW copy (scale = 1). No stats read, no LSE -- correction depends
                            # only on the mma (o_acc) and the sO ring.
                            scale = Float32(1.0)
                            pipeline_o_acc.consumer_wait_w_index_phase(stage, o_phase)
                            # Rolling drain: thread b's group from o_buffers tiles ago read THIS
                            # buffer; wait_group(o_buffers-1) proves it done. The WG barrier
                            # publishes that to all 128 evac threads. INLINE (no closure: captures
                            # through nested dynamic regions mis-resolve).
                            if tidx < const_expr(self.o_nbox):
                                cute.arch.cp_async_bulk_wait_group(const_expr(self.o_buffers - 1), read=True)
                            corr_bar.arrive_and_wait()
                            # Normalize + evacuate O[stage] from tmem to sO buffer `buf`, via its
                            # merged [m_block, D] view of the [qhead, D, o_nbox] tile (dynamic
                            # buffer slice; layout shared across buffers).
                            sO_evac_b = cute.make_tensor(sO[None, None, None, buf].iterator, sO_evac_layout)
                            self.correction_epilogue(
                                thr_mma_pv,
                                tOtO[None, None, None, stage],
                                tidx,
                                stage,
                                kv_block,
                                Int32(self.m_block_size),
                                scale,
                                sO_evac_b,
                                None,
                                None,
                                None,
                            )
                            # O tmem buffer now read; release it back to the mma (o_acc empty).
                            pipeline_o_acc.consumer_release_w_index(stage)
                            # Publish all 128 threads' evac writes (each fenced by
                            # correction_epilogue) to the issuing threads, then thread b issues
                            # box b's bulk-TMA + commit (drain happens at the NEXT rotation).
                            # Reuses _store_O (dest math + copy + commit). INLINE (no closure;
                            # see drain note above).
                            corr_bar.arrive_and_wait()
                            bSG_sO_c, bSG_gO_c = cpasync.tma_partition(
                                tma_atom_O,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sO[None, None, None, buf], 0, 2),
                                gO_grp_c,
                            )
                            self._store_O(
                                tma_atom_O,
                                bSG_sO_c,
                                bSG_gO_c,
                                hkv,
                                base,
                                row_start,
                                n_rows,
                                tidx,
                                tqk,
                            )
                            so_state.advance()
                        # Per-group phase flips: o_acc and sm_stats are indexed by stage, so each
                        # stage's barrier advances once per group (sO is handled by so_state).
                        o_phase ^= 1
            work_tile = tile_scheduler.advance_to_next_work()
        # Tail: outstanding bulk-TMA reads must finish before CTA teardown.
        if tidx < const_expr(self.o_nbox):
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.kernel
    def kernel(
        self,
        mQ,
        mK,
        mV,
        mO,
        mM,
        mL,
        mTopkSlotIds,
        mKvToQOffsets,
        mKvToQIdxRank,
        mWorkStart,
        mWorkEnd,
        mSelSlots,
        mNumSel,
        aux_tensors,
        mCuSeqlensQ,
        mSeqUsedK,
        n_batches,
        tma_atom_K,
        tma_atom_V,
        tma_atom_Q,
        mQ_tma,
        softmax_scale_log2,
        softmax_scale,
        sQ_layout,
        sK_layout,
        tP_layout,
        sV_layout,
        sO_layout,
        tiled_mma_qk,
        tiled_mma_pv,
        tile_sched_params,
        tma_atom_O=None,
        mO_tma=None,
        gmem_tiled_copy_Q=None,
    ):
        # Flat-block decode divisor: nbs = num_block_slots (the index tensors' block dim).
        nbs = Int32(mKvToQOffsets.shape[1] - 1)
        # Compact selected-slot tensors, stashed on self so the inlined warp helpers
        # (_slice_block / _next_cj / _num_real_cj) read them without threading through every
        # warp-function signature. mKvToQOffsets here is the COMPACT CSR (sel_offsets).
        self._mSelSlots = mSelSlots
        self._mNumSel = mNumSel
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (tma_atom_K, tma_atom_V, tma_atom_Q, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)
        cta_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.softmax0_warp_ids, *self.softmax1_warp_ids, *self.correction_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        TCG = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = TCG(len([self.mma_warp_id]))
        correction_threads = TCG(cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        sm_cluster = TCG(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids) * self.cta_group_size)
        corr_cluster = TCG(cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size)

        # Q is gathered by the 3 cooperative load warps via cp.async (the freed store warp
        # joins the two load warps = 96 producer threads), so its pipeline is armed via
        # cp.async mbarrier arrivals rather than a TMA tx-count.
        pipeline_q = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_load_stage,
            producer_group=TCG(cute.arch.WARP_SIZE * (len(self.load_warp_ids) + 1)),
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=TCG(1),
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # s_p: mma -> softmax. mma commits S-full (after QK) + acquires P-full (before PV);
        # softmax waits S-full + releases P-full. Consumer = one softmax warpgroup per stage.
        pipeline_s_p = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_s_p.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=sm_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # o_acc: the O tmem buffer between mma (producer) and correction (consumer), bidirectional.
        # mma producer_acquire (O buffer free = correction done reading prior O; the former s_o
        # "O rescaled" back-edge) before PV, then producer_commit (O full) after. correction
        # consumer_wait (O full) then consumer_release (O free) after the evac. The empty barrier is
        # producer-pre-armed at init, so no consumer pre-release is needed.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=corr_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # (The former sScale/pipeline_sm_stats softmax->correction stats handoff is GONE:
        # softmax exports (m~, l) straight to gmem and the combine normalizes O.)
        # sO buffer handoff: correction (producer; acquire empty -> evac -> commit full) to the
        # store warp (consumer; wait full -> bulk-TMA issue + drain -> release empty). Stages =
        # o_buffers (buffer == softmax stage when 2; single serialized buffer when 1).
        store_warp_threads = TCG(cute.arch.WARP_SIZE)
        pipeline_sO = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_sO.data_ptr(),
            num_stages=self.o_buffers,
            producer_group=correction_threads,
            consumer_group=store_warp_threads,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        # sO multi-buffered: append an o_buffers buffer mode (stride = one-tile cosize) above the
        # epi swizzle bits. sO[..., b] is the b-th [qhead, D, o_nbox] tile buffer.
        sO_tile_cosize = const_expr(cute.cosize(sO_layout))
        sO = storage.sO.get_tensor(
            cute.append(sO_layout.outer, cute.make_layout(self.o_buffers, stride=sO_tile_cosize)),
            swizzle=sO_layout.inner,
        )
        # (slot, box, qidx|rank) ring of each in-flight tile's per-box Q indices (see
        # _ring_depths), and the (slot, row, m~|l) softmax-stats ring for the store warp.
        nbox_m = const_expr(self.m_block_size // self.qhead_per_kvhead)
        sQIdxRank = storage.sQIdxRank.get_tensor(
            cute.make_ordered_layout((self.pairs_depth, nbox_m, 2), order=(2, 1, 0))
        )

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        tP_width_ratio = Float32.width // self.v_dtype.width
        tP_stage_stride = (self.tmem_p_offset[1] - self.tmem_p_offset[0]) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(tOrP.layout, cute.make_layout((self.s_stage,), stride=(tP_stage_stride,))),
        )
        block_info = BlockInfo(
            self.cta_tiler[0],
            self.cta_tiler[1],
            False,
            False,
            False,
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[1],
            seqlen_k_static=self.block_size,
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )
        from flash_attn.cute.mask import AttentionMask

        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=None,
            window_size_right=None,
            qhead_per_kvhead_packgqa=1,
        )
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
        tile_scheduler = StaticPersistentTileScheduler.create(tile_sched_params)

        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
        # The store warp (15) joins the two load warps for the cooperative cp.async Q gather.
        load_hi = const_expr(self.store_warp_id)
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= load_hi:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_K,
                tma_atom_V,
                tma_atom_Q,
                mQ_tma,
                pipeline_q,
                pipeline_kv,
                thr_mma_qk,
                thr_mma_pv,
                mTopkSlotIds,
                mKvToQOffsets,
                mKvToQIdxRank,
                mWorkStart,
                mWorkEnd,
                nbs,
                mSeqUsedK,
                n_batches,
                sQIdxRank,
                tile_scheduler=tile_scheduler,
                gmem_tiled_copy_Q=gmem_tiled_copy_Q,
            )
        if warp_idx <= self.mma_warp_id:
            if warp_idx == self.mma_warp_id:
                tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            # All TMEM participants must rendezvous at one generated barrier instruction.
            # Calling wait_for_alloc from role-specific branches creates separate PCs for the
            # same named barrier and is reported as divergent by synccheck.
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.mma(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    sQ,
                    sK,
                    sV,
                    tStS,
                    tOtO,
                    tOrP,
                    pipeline_q,
                    pipeline_kv,
                    pipeline_s_p,
                    pipeline_o_acc,
                    mKvToQOffsets,
                    mWorkStart,
                    mWorkEnd,
                    nbs,
                    mSeqUsedK,
                    n_batches,
                    tile_scheduler=tile_scheduler,
                )
                tmem.relinquish_alloc_permit()
            if warp_idx <= self.softmax1_warp_ids[-1]:
                # softmax0 (warps 0-3) handles stage 0, softmax1 (warps 4-7) stage 1 (ping-pong).
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
                stage = Int32(0) if warp_idx < self.softmax1_warp_ids[0] else Int32(1)
                self.softmax_loop(
                    stage,
                    softmax_scale_log2,
                    softmax_scale,
                    thr_mma_qk,
                    tStS,
                    mM,
                    mL,
                    sQIdxRank,
                    pipeline_s_p,
                    block_info,
                    SeqlenInfoCls,
                    AttentionMaskCls,
                    aux_tensors,
                    mKvToQOffsets,
                    mKvToQIdxRank,
                    mCuSeqlensQ,
                    mSeqUsedK,
                    n_batches,
                    mWorkStart,
                    mWorkEnd,
                    nbs,
                    Int32(mQ.shape[0]),
                    tile_scheduler=tile_scheduler,
                )
            if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
                self.correction_loop(
                    thr_mma_qk,
                    thr_mma_pv,
                    tStS,
                    tOtO,
                    sO,
                    pipeline_o_acc,
                    pipeline_sO,
                    mKvToQOffsets,
                    mWorkStart,
                    mWorkEnd,
                    nbs,
                    mSeqUsedK,
                    n_batches,
                    tile_scheduler=tile_scheduler,
                    tma_atom_O=tma_atom_O,
                    mO_tma=mO_tma,
                    tqk=Int32(mM.shape[1]) // const_expr(self.qhead_per_kvhead),
                )
            tmem_alloc_barrier.arrive_and_wait()
            if warp_idx == self.mma_warp_id:
                cute.arch.dealloc_tmem(
                    tmem_ptr,
                    cute.arch.get_max_tmem_alloc_cols("sm_100"),
                    is_two_cta=False,
                    arch="sm_100",
                )
        return

    @cute.jit
    def __call__(
        self,
        mQ,
        mK,
        mV,
        mO,
        mM,
        mL,
        mTopkSlotIds,
        mKvToQOffsets,
        mKvToQIdxRank,
        mWorkStart,
        mWorkEnd,
        mSelSlots,
        mNumSel,
        grid_size: Int32,
        mCuSeqlensQ,
        mSeqUsedK,
        n_batches: Int32,
        softmax_scale: Float32,
        mO2d=None,
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[1, 3, 2, 0]))
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[1, 0, 2, 3]))
        self._setup_attributes()
        # One K + one V buffer per block (the whole 128-key block is resident for one gemm).
        # page_size<128 fills each buffer with `ratio` page-sized TMAs (see _load_block_paged),
        # so kv_stage stays 2 regardless of ratio. Deeper buffering was measured to regress:
        # the extra sK smem cuts CTA occupancy by more than the K/V prefetch gains, since the
        # kernel is smem-heavy (fp32 sO).
        self.kv_stage = 2
        # One Q buffer per stage (the two 128-row halves of a work-item are both resident
        # so the two softmax warpgroups can run concurrently).
        self.q_load_stage = self._q_load_stage_cfg
        self.use_tma_O = False
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 1)
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = self._tune.get("ex2_emu_freq", 16)

        cta_group = tcgen05.CtaGroup.ONE
        self.o_layout = cutlass.utils.LayoutEnum.ROW_MAJOR
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            tcgen05.OperandSource.TMEM,
        )
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)
        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_load_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage)
        tP_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage)
        sV_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage)
        # O_partial store: sO is o_nbox independently-swizzled [qhead, D] boxes (same total bytes
        # as one [m_block, D] tile, so smem use is unchanged). The correction evac fills sO through
        # a merged [m_block, D] view (row r -> box r // qhead, so box b == packed rows
        # [b*qhead, (b+1)*qhead) == one query's qhead-group); each box is then stored with one
        # [qhead, D] bulk-TMA to a scattered, box-aligned O_partial row group. make_tiled_tma_atom
        # auto-derives the epi swizzle (dense_gemm idiom). epi_tile stays (m_block, head_dim_v) so
        # the base correction_epilogue's tmem->smem evac (Ld32x32b, conflict-free) is unchanged.
        # o_nbox: [qhead, D] boxes per 128-row O tile (= 8). NOT a pipeline depth (that is
        # o_buffers); the name mirrors FA4's epi "stage" machinery it is built with.
        self.o_nbox = const_expr(self.m_block_size // self.qhead_per_kvhead)
        o_epi_tile = (self.qhead_per_kvhead, self.head_dim_v_padded)
        sO_layout = sm100_utils_basic.make_smem_layout_epi(self.o_dtype, self.o_layout, o_epi_tile, self.o_nbox)
        tma_atom_O, mO_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO2d, cute.slice_(sO_layout, (None, None, 0)), o_epi_tile
        )
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [("K", mK, sK_layout), ("V", mV, sV_layout)]
        }
        # Whole-tile Q tx-count: the nbox_m x n_kblk [qhead, K_ATOM] box copies of one
        # 128-row tile all land on one stage barrier.
        self.tma_copy_bytes["Q"] = self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        # Q TMA: [qhead, K_ATOM] K-half boxes over the flat [Tq*Hq, D] Q view, landing in the
        # MMA-A swizzle (one swizzle atom per box; see prototype/bench_tma_q_load.py). A box is
        # one query's contiguous qhead-group of rows, so the packed gather needs one qidx per
        # box instead of one per row.
        k_atom = 1024 // self.q_dtype.width
        q_box_tile = (self.qhead_per_kvhead, k_atom)
        q_box_layout = cute.slice_(
            sm100_utils_basic.make_smem_layout(tcgen05.OperandMajorMode.K, q_box_tile, self.q_dtype, 1),
            (None, None, 0),
        )
        mQ2d = cute.make_tensor(
            mQ.iterator,
            cute.make_layout((mQ.shape[0] * mQ.shape[1], mQ.shape[2]), stride=(mQ.shape[2], 1)),
        )
        # Cooperative cp.async Q-gather tiled copy (store-in-corr path): 3 load warps
        # (96 threads) cover a [m_block, D] tile; 128-bit loads, GLOBAL cache mode.
        sic_load_threads = cute.arch.WARP_SIZE * (len(self.load_warp_ids) + 1)
        sic_async_elems = 128 // self.q_dtype.width
        sic_tpr = math.gcd(self.head_dim_padded // sic_async_elems, sic_load_threads)
        gmem_tiled_copy_Q = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self.q_dtype,
                num_bits_per_copy=sic_async_elems * self.q_dtype.width,
            ),
            cute.make_ordered_layout((sic_load_threads // sic_tpr, sic_tpr), order=(1, 0)),
            cute.make_layout((1, sic_async_elems)),
        )
        tma_atom_Q, mQ_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ2d, q_box_layout, q_box_tile
        )
        if const_expr(self.ratio == 1):
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk.shape,
            )
        else:
            # page_size < 128: page-sized K/V TMA atoms. The atom's smem box is a
            # swizzle-preserving page stripe sliced from the live 128-key smem-B layout (a
            # fresh page-sized layout would pick a different swizzle and scramble the 128-key
            # MMA read). K is paged on N (=keys) -> build a page-N tiled_mma just for the atom
            # (the consuming QK MMA stays 128-wide); V is paged on the contraction (=keys) ->
            # the full PV mma's V-map (over head_dim_v) is unaffected (the hd512 Vt pattern).
            # tma_copy_bytes stays the full-128 value: the ratio page copies sum to it.
            ps = self.page_size
            mma_tiler_qk_pg = (self.mma_tiler_qk[0], ps, self.mma_tiler_qk[2])
            mma_tiler_pv_pg = (self.mma_tiler_pv[0], self.mma_tiler_pv[1], ps)
            tiled_mma_qk_pg = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.qk_acc_dtype,
                cta_group,
                mma_tiler_qk_pg[:2],
            )
            k_box = tiled_mma_qk_pg.partition_shape_B(cute.dice(mma_tiler_qk_pg, (None, 1, 1)))
            sK_box = cute.select(cute.flat_divide(sK_layout, k_box), mode=[0, 1, 2])
            v_per = cute.size(sV_layout, mode=[2]) // self.ratio
            sV_box = cute.select(
                cute.flat_divide(sV_layout, (cute.size(sV_layout, mode=[0]), 1, v_per)),
                mode=[0, 1, 2],
            )
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                sK_box,
                mma_tiler_qk_pg,
                tiled_mma_qk_pg,
                cta_layout_vmnk.shape,
            )
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                sV_box,
                mma_tiler_pv_pg,
                tiled_mma_pv,
                cta_layout_vmnk.shape,
            )
        # Load-balanced KV-stationary: one work-item per scheduler split (grid_size of them),
        # each a contiguous run of the global (kv_head, kv_block, query) work sequence given by
        # mWorkStart/mWorkEnd. The tile scheduler just enumerates wi in [0, grid_size); the
        # block run + per-block Q-tile loop are decoded in-kernel. Sentinel work-items
        # (past the real work) are skipped (num_fb==0).
        tile_sched_args = TileSchedulerArguments(
            num_block=grid_size,
            num_head=Int32(1),
            num_batch=Int32(1),
            num_splits=Int32(1),
            seqlen_k=self.block_size,
            headdim=self.head_dim_padded,
            headdim_v=self.head_dim_v_padded,
            total_q=grid_size * self.m_block_size,
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=True,
            lpt=False,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=False,
        )
        tile_sched_params = StaticPersistentTileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=SchedulingMode.STATIC
        )
        grid_dim = StaticPersistentTileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)

        # sO is multi-buffered for bulk-TMA O_partial: buffers alternate across packed-row
        # groups (g) so a group's store can overlap the next group's tmem->smem evac. Falls
        # back to 1 if smem budget exceeded.
        def _make_shared_storage(o_buffers):
            pairs_depth, stats_depth = self._ring_depths(o_buffers)
            nbox_m = self.m_block_size // self.qhead_per_kvhead

            @cute.struct
            class SharedStorage:
                mbar_load_Q: cute.struct.MemRange[Int64, self.q_load_stage * 2]
                mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
                # s_p carries S-full (->softmax) + P-full (softmax->mma). The O tmem buffer's
                # full (mma->correction) AND empty/free (correction->mma) edges both live on
                # mbar_O_full (pipeline_o_acc), so no separate s_o barrier is needed.
                mbar_s_p: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
                # sO buffer full/empty handoff between correction and the store warp.
                mbar_sO: cute.struct.MemRange[Int64, o_buffers * 2]
                tmem_dealloc_mbar_ptr: Int64
                tmem_holding_buf: Int32
                # qidx/rank pairs ring: pairs_depth slots x nbox_m boxes x (qidx, rank).
                # (Write-only today; retained for the causal-mask smem consumer, sec 1f.)
                sQIdxRank: cute.struct.MemRange[Int32, pairs_depth * nbox_m * 2]
                sO: cute.struct.Align[cute.struct.MemRange[self.o_dtype, sO_size * o_buffers], self.buffer_align_bytes]
                sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)], self.buffer_align_bytes
                ]

            return SharedStorage

        smem_cap = getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()), "shared_memory_per_block_optin", 232448
        )
        self.o_buffers = self._o_buffers_cfg
        SharedStorage = _make_shared_storage(self.o_buffers)
        # Step down (not collapse) on smem pressure: each fewer buffer frees one sO tile.
        # (Chained const_expr ifs: the DSL AST rejects closures inside loop constructs.)
        if const_expr(self.o_buffers > 1 and SharedStorage.size_in_bytes() > smem_cap):
            self.o_buffers -= 1
            SharedStorage = _make_shared_storage(self.o_buffers)
        if const_expr(self.o_buffers > 1 and SharedStorage.size_in_bytes() > smem_cap):
            self.o_buffers -= 1
            SharedStorage = _make_shared_storage(self.o_buffers)
        if const_expr(self.o_buffers > 1 and SharedStorage.size_in_bytes() > smem_cap):
            self.o_buffers -= 1
            SharedStorage = _make_shared_storage(self.o_buffers)
        self.pairs_depth, self.stats_depth = self._ring_depths(self.o_buffers)
        self.shared_storage = SharedStorage
        smem_size = SharedStorage.size_in_bytes()
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, None)
        # Masking is handled by the bespoke per-tile _apply_mask, computed entirely in-kernel
        # from cu_seqlens_q + used_kv_lens (real per-seq Lk_b); not the mask_mod hook.
        aux_tensors = None

        _enable_kvouter_debug_artifacts()
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mM,
            mL,
            mTopkSlotIds,
            mKvToQOffsets,
            mKvToQIdxRank,
            mWorkStart,
            mWorkEnd,
            mSelSlots,
            mNumSel,
            aux_tensors,
            mCuSeqlensQ,
            mSeqUsedK,
            n_batches,
            tma_atom_K,
            tma_atom_V,
            tma_atom_Q,
            mQ_tma,
            softmax_scale_log2,
            softmax_scale,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            tma_atom_O,
            mO_tma,
            gmem_tiled_copy_Q,
        ).launch(grid=grid_dim, block=[self.threads_per_cta, 1, 1], smem=smem_size, stream=stream, min_blocks_per_mp=1)


_NEG_INF = float("-inf")
_compile_cache: dict = {}


def _build_inverse_map(
    kv_to_q_offsets: torch.Tensor,
    kv_to_q_indices_and_ranks: torch.Tensor,
    hkv_n: int,
    tq: int,
    topk: int,
) -> torch.Tensor:
    """``inv[hkv, q, rank] = pair position p`` (-1 where never materialized, e.g.
    causal-clipped) -- the combine's gather map for the tile-ordered flat partials.

    PROTOTYPE host build, fully SYNC-FREE (no boolean indexing / nonzero, no D2H): invalid
    tail positions are redirected to a sacrificial extra slot via ``torch.where``. In
    the index builder can implement this as one extra coalesced scatter store,
    making the cost zero."""
    device = kv_to_q_offsets.device
    P = kv_to_q_indices_and_ranks.shape[1]
    total = kv_to_q_offsets[:, -1:].to(torch.int64)  # [hkv, 1] valid pair counts
    pos = torch.arange(P, device=device, dtype=torch.int32).unsqueeze(0).expand(hkv_n, P)
    valid = pos.to(torch.int64) < total
    qq = kv_to_q_indices_and_ranks[..., 0].to(torch.int64).clamp_(0, tq - 1)
    rr = kv_to_q_indices_and_ranks[..., 1].to(torch.int64).clamp_(0, topk - 1)
    hh = torch.arange(hkv_n, device=device, dtype=torch.int64).unsqueeze(1)
    dummy = hkv_n * tq * topk  # one-past-end slot absorbs all invalid positions
    flat_dst = torch.where(valid, (hh * tq + qq) * topk + rr, dummy)
    inv_flat = torch.full((hkv_n * tq * topk + 1,), -1, dtype=torch.int32, device=device)
    inv_flat.scatter_(0, flat_dst.reshape(-1), pos.reshape(-1))
    return inv_flat[:-1].view(hkv_n, tq, topk)


def _enable_kvouter_debug_artifacts():
    """Dump PTX/CUBIN with source line info for this JIT, gated by
    ``MINIMAX_KERNELS_KVOUTER_DEBUG_ARTIFACTS=1`` (debug-only; default off -- the dumps land in the
    process CWD and lineinfo changes codegen). Only effective at first compile."""
    if os.environ.get("MINIMAX_KERNELS_KVOUTER_DEBUG_ARTIFACTS", "0") != "1":
        return
    dsl = CuTeDSL()
    dsl.envar.keep_ptx = True
    dsl.envar.keep_cubin = True
    dsl.envar.lineinfo = True


def _get_compiled(qhead_per_kvhead, nheads_kv, page_size, causal, q_load_stage, o_buffers, templates):
    q_t, _, _, o_t, *_ = templates
    # n_batches is a dynamic Int32 kernel arg (like grid_size), NOT a compile key: one kernel
    # serves any batch count, so varying batch sizes (incl. B=1) never trigger a recompile.
    key = (
        qhead_per_kvhead,
        nheads_kv,
        page_size,
        causal,
        q_load_stage,
        o_buffers,
        q_t.element_type,
        o_t.element_type,
    )
    if key not in _compile_cache:
        kernel = SparseKVOuterForward(
            qhead_per_kvhead,
            nheads_kv,
            page_size,
            causal=causal,
            q_load_stage=q_load_stage,
            o_buffers=o_buffers,
        )
        (
            q_t,
            k_t,
            v_t,
            o_t,
            m_t,
            l_t,
            slot_t,
            off_t,
            ir_t,
            ws_t,
            we_t,
            ss_t,
            ns_t,
            gs,
            cuq_t,
            sk_t,
            nb,
            scale,
            o2d_t,
        ) = templates
        _enable_kvouter_debug_artifacts()
        _compile_cache[key] = cute.compile(
            kernel,
            q_t,
            k_t,
            v_t,
            o_t,
            m_t,
            l_t,
            slot_t,
            off_t,
            ir_t,
            ws_t,
            we_t,
            ss_t,
            ns_t,
            gs,
            cuq_t,
            sk_t,
            nb,
            scale,
            o2d_t,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _compile_cache[key]


def _arch_defaults(q_dtype: torch.dtype, partial_dtype: torch.dtype) -> tuple:
    """Tuned (q_load_stage, o_buffers) defaults for the store-in-correction arch (B200
    sweeps, fp8 q4k/kv60k): qls=4 / obuf=4 (7ae sweep optimum, 74.9 us).

    16-bit inputs halve both depths (sQ/sO double in bytes; deeper rings exceed the smem
    budget and the step-down would collapse the sO ring). fp32 partials force a single sO
    buffer. o_buffers is capped at 4 (o_buffers x o_nbox issuing lane sets <= 32 lanes)."""
    is_fp8 = q_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    q_load_stage = 4 if is_fp8 else 2
    o_buffers = 4 if is_fp8 else 2
    if partial_dtype == torch.float32:
        o_buffers = 1
    return q_load_stage, o_buffers


@lru_cache(maxsize=None)
def _cached_sm_count(device_index: int) -> int:
    # multi_processor_count is static per device; cache to avoid a
    # get_device_properties (-> _get_device_index) call on every prefill.
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def indexed_block_partials(
    q,
    k_cache,
    v_cache,
    topk_slot_ids,
    kv_to_q_offsets,
    kv_to_q_indices_and_ranks,
    *,
    topk,
    block_size,
    page_size,
    softmax_scale,
    causal=False,
    cu_seqlens_q=None,
    used_kv_lens=None,
    q_load_stage=None,
    partial_dtype: Optional[torch.dtype] = None,
    inv: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    sel_slots: Optional[torch.Tensor] = None,
    sel_offsets: Optional[torch.Tensor] = None,
    num_sel: Optional[torch.Tensor] = None,
):
    """KV-outer / Q-inner producer (store-in-correction arch: O_partial stored from the
    correction warpgroup with a 3-warp cooperative cp.async Q gather).

    ``q`` is token-major [Tq,Hq,D] and ``k_cache``/``v_cache`` use the
    cache layout [num_pages,Hkv,page_size,D]. Returns tile-ordered flat partials
    (o_flat [Hkv*Tq*topK*qhead, D],
    lse [Hq,Tq,topK]).

    ``partial_dtype`` controls ``O_partial`` storage (default ``q.dtype`` for
    bandwidth; ``torch.float32`` for correctness tests). LSE is always fp32.

    Masking is computed entirely in-kernel from ``cu_seqlens_q`` + ``used_kv_lens`` (no
    host-precomputed causal tensor / positions). Pack all batches' queries into
    ``Tq = total_q``, group the KV block-slots per batch in the index tensors (block-slot
    index == global KV-block index, batch-contiguous), and pass ``cu_seqlens_q`` [B+1]
    torch.int64 on device. For a single sequence, callers must pass the B=1 metadata
    ``[0, Tq]``. ``used_kv_lens`` [B] int32 is the REAL per-sequence KV length ``Lk_b``
    (supports variable / non-128-multiple lengths): causal masks key pos <= query pos
    ``(t - q_off) + (Lk_b - Tq_b)`` (right-aligned suffix); non-causal masks key pos
    < ``Lk_b`` (drops the partial last block's padding).
    """
    assert block_size == 128 and q.shape[-1] == 128
    assert page_size in (64, 128), "page_size must be 64 or 128"
    # page_size=64 (ratio=2) loads each 128-key block with ratio page-sized TMAs into one
    # smem buffer (see load()/_load_block_paged). topk_slot_ids must hold num_block_slots*ratio
    # physical-page slots per head (block kv_block -> slots [kv_block*ratio, +ratio)).
    tq, hq, d = q.shape
    num_pages, nheads_kv, ps, dv = k_cache.shape
    qhead = hq // nheads_kv
    device = q.device
    _FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
    q_is_fp8 = q.dtype in _FP8_DTYPES
    if q_is_fp8:
        # Pure-fp8 path (no scaling): Q/K/V are fp8, QK & PV run fp8 tcgen05 MMAs (fp32 acc),
        # softmax in fp32, P cast to fp8 for PV. O_partial stays bf16 (below). K/V must share Q's
        # fp8 dtype (single MMA ab_dtype).
        assert k_cache.dtype == q.dtype and v_cache.dtype == q.dtype, (
            f"fp8 path requires k/v_cache dtype == q dtype ({q.dtype}); " f"got k={k_cache.dtype}, v={v_cache.dtype}"
        )
    if partial_dtype is None:
        # fp8 inputs keep O_partial in bf16 (fp8 partials would lose too much precision); 16-bit
        # inputs default to q.dtype for store bandwidth.
        partial_dtype = torch.bfloat16 if q_is_fp8 else q.dtype
    assert partial_dtype in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ), f"partial_dtype must be fp32/bf16/fp16, got {partial_dtype}"
    # TILE-ORDERED outputs: O_partial and the (m~, l) stats are stored by PAIR
    # POSITION p = base + ql (the forward's natural processing order), not by (q, rank):
    #   o_flat [Hkv * Tq*topK * qhead, D]  row (hkv, p, hl) -- a tile writes 128 CONSECUTIVE
    #                                      rows, so the store warp needs no index gathers;
    #   m/l_flat [Hkv, Tq*topK * qhead]    softmax exports two coalesced 4B stores per row.
    # The combine resolves (q, rank) -> p via the inverse map below. Shapes are FIXED across
    # requests (per-head segment bound p < Tq*topK is exact); unused tail rows are never
    # written nor read, so plain torch.empty (no init fills) suffices -- no D2H anywhere.
    hkv_n = nheads_kv
    qhead = hq // hkv_n
    seg = tq * topk * qhead
    o_flat = torch.empty(hkv_n * tq * topk * qhead, d, dtype=partial_dtype, device=device)
    m_partial = torch.empty((hkv_n, seg), dtype=torch.float32, device=device)
    l_partial = torch.empty((hkv_n, seg), dtype=torch.float32, device=device)
    # The inverse map is a property of the INDEX, not the forward: callers that reuse a
    # selection build it once alongside the index and pass it in; building
    # it here per call costs ~8 small kernels (prototype convenience for tests).
    if inv is None:
        inv = _build_inverse_map(kv_to_q_offsets, kv_to_q_indices_and_ranks, hkv_n, tq, topk)

    # Load-balanced KV-stationary: a device-side scheduler (no D2H) partitions the global
    # (kv_head, kv_block, query) work into grid_size = NUM_SPLITS balanced runs; each kernel
    # work-item processes one run (possibly spanning many small blocks or a slice of a large
    # one). NUM_SPLITS defaults to one persistent CTA wave; extra waves add
    # prologue/drain overhead after the scheduler has already balanced the work.
    # The Q-tile count per block is an in-kernel loop; sentinel (past-the-work) splits
    # are skipped in-kernel.
    if num_splits is None:
        num_splits = _cached_sm_count(device.index if device.index is not None else torch.cuda.current_device())
    # The forward iterates the COMPACT selected-slot index (only selected blocks), skipping both
    # msb-padding and unselected real blocks. The scheduler is fed sel_offsets (plateaued at the
    # head total), so it emits COMPACT-j runs and its head_base logic is unchanged. The compact
    # index is REQUIRED and is fused into CountToOffsets by build_kvouter_index.
    assert (
        sel_slots is not None and sel_offsets is not None and num_sel is not None
    ), "indexed_block_partials requires the compact index (sel_slots/sel_offsets/num_sel)"
    work_start, work_end, grid_size, _nqps = build_load_balanced_schedule(
        sel_offsets, total_q=tq, num_splits=num_splits, topk=topk
    )
    # Per-arch tuned pipeline depths (see _arch_defaults); smem pressure steps o_buffers
    # down inside __call__. q_load_stage > 2 lets the load warps run ahead of the MMA to
    # hide Q-gather latency, at the cost of sQ smem.
    qls_default, o_buffers = _arch_defaults(q.dtype, partial_dtype)
    if q_load_stage is None:
        q_load_stage = qls_default
    assert q_load_stage >= 2, "q_load_stage must be >= 2 (in-flight ping-pong tiles)"

    k_perm = k_cache.permute(0, 2, 1, 3)
    v_perm = v_cache.permute(0, 2, 1, 3)
    assert cu_seqlens_q is not None, "cu_seqlens_q is required; use [0, Tq] for single sequence"
    assert (
        cu_seqlens_q.dtype == torch.int64 and cu_seqlens_q.is_contiguous()
    ), "cu_seqlens_q must be contiguous torch.int64"
    mCuSeqlensQ = cu_seqlens_q
    n_batches = mCuSeqlensQ.shape[0] - 1
    # Real per-seq KV length Lk_b drives the in-kernel mask (causal suffix limit + non-causal
    # padding). Default (None) reproduces the legacy uniform assumption Lk_b = msb*block_size
    # (msb = num_block_slots // B); pass an explicit [B] tensor for variable / non-128-multiple
    # KV lengths. The public interface forwards explicit real sequence lengths.
    nbs_host = kv_to_q_offsets.shape[1] - 1
    if used_kv_lens is None:
        msb = nbs_host // n_batches
        mSeqUsedK = torch.full((n_batches,), msb * block_size, dtype=torch.int32, device=device)
    else:
        mSeqUsedK = used_kv_lens.to(device=device, dtype=torch.int32).contiguous()
        assert (
            mSeqUsedK.shape[0] == n_batches
        ), f"used_kv_lens length ({mSeqUsedK.shape[0]}) must equal n_batches ({n_batches})"

    q_t = to_cute_tensor(q, leading_dim=2)
    k_t = to_cute_tensor(k_perm, leading_dim=3)
    v_t = to_cute_tensor(v_perm, leading_dim=3)
    m_t = to_cute_tensor(m_partial, assumed_align=4, leading_dim=1)
    l_t = to_cute_tensor(l_partial, assumed_align=4, leading_dim=1)
    slot_t = to_cute_tensor(topk_slot_ids, assumed_align=8, leading_dim=1)
    off_t = to_cute_tensor(sel_offsets, assumed_align=4, leading_dim=1)  # COMPACT CSR (mKvToQOffsets)
    ir_t = to_cute_tensor(kv_to_q_indices_and_ranks, assumed_align=4, leading_dim=2)
    ws_t = to_cute_tensor(work_start, assumed_align=4, leading_dim=1)
    we_t = to_cute_tensor(work_end, assumed_align=4, leading_dim=1)
    ss_t = to_cute_tensor(sel_slots, assumed_align=4, leading_dim=1)
    ns_t = to_cute_tensor(num_sel, assumed_align=4, leading_dim=0)
    gs = int(grid_size)
    cuq_t = to_cute_tensor(mCuSeqlensQ, assumed_align=8, leading_dim=0)
    sk_t = to_cute_tensor(mSeqUsedK, assumed_align=4, leading_dim=0)
    nb = int(n_batches)
    scale = float(softmax_scale)
    # The [qhead, D] bulk-TMA store atom is built over the flat [total_rows, D] O_partial;
    # a box's dest row group is simply hkv*Tq*topK + base + ql (tile-ordered).
    o2d = o_flat
    o2d_t = to_cute_tensor(o2d, leading_dim=1)
    o_t = o2d_t  # mO is only used for o_dtype; the flat 2D tensor serves both roles
    compiled = _get_compiled(
        qhead,
        nheads_kv,
        page_size,
        bool(causal),
        int(q_load_stage),
        o_buffers,
        (q_t, k_t, v_t, o_t, m_t, l_t, slot_t, off_t, ir_t, ws_t, we_t, ss_t, ns_t, gs, cuq_t, sk_t, nb, scale, o2d_t),
    )
    # fp8 cute tensors lower to a uint8 ABI param, but the tvm-ffi runtime dtype check rejects
    # torch.float8_*; pass byte-identical uint8 views (same 1-byte itemsize, strides preserved)
    # while the compiled kernel still treats the data as fp8 (template element_type is fp8).
    q_rt = q.view(torch.uint8) if q_is_fp8 else q
    k_rt = k_perm.view(torch.uint8) if q_is_fp8 else k_perm
    v_rt = v_perm.view(torch.uint8) if q_is_fp8 else v_perm
    compiled(
        q_rt,
        k_rt,
        v_rt,
        o_flat,
        m_partial,
        l_partial,
        topk_slot_ids,
        sel_offsets,
        kv_to_q_indices_and_ranks,
        work_start,
        work_end,
        sel_slots,
        num_sel,
        gs,
        mCuSeqlensQ,
        mSeqUsedK,
        nb,
        scale,
        o2d,
    )
    return o_flat, m_partial, l_partial, inv


def sparse_kvouter_attn_fwd_indexed(
    q: torch.Tensor,  # [Tq, Hq, D]
    k_cache: torch.Tensor,  # [num_pages, Hkv, page_size, D]
    v_cache: torch.Tensor,
    topk_slot_ids: torch.Tensor,  # [Hkv, num_block_slots * ratio] int64
    kv_to_q_offsets: torch.Tensor,  # [Hkv, num_block_slots + 1] int32
    kv_to_q_indices_and_ranks: torch.Tensor,  # [Hkv, Tq*topK, 2] int32
    *,
    topk: int,
    block_size: int = 128,
    page_size: int = 64,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    used_kv_lens: Optional[torch.Tensor] = None,
    q_load_stage: Optional[int] = None,
    partial_dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    return_lse: bool = False,
    inv: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    sel_slots: Optional[torch.Tensor] = None,
    sel_offsets: Optional[torch.Tensor] = None,
    num_sel: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KV-outer attention over a paged cache + pre-built index tensors (forward + merge).

    Produces per-(q, rank) partials with the KV-stationary kernel (``indexed_block_partials``)
    then log-sum-exp-combines them across ranks (``merge_kv_partials``). Returns
    ``(o [Tq, Hq, D], lse [Hq, Tq] or None)`` (head-major LSE, the FA-forward convention).
    This is the indexed building block; the high-level :func:`...interface.kvouter_attention`
    builds the index then calls this.

    Pack all batches' queries into ``Tq = total_q`` and group KV block-slots per batch in the
    index tensors. ``cu_seqlens_q`` is required ([B+1] contiguous torch.int64, device); use
    ``[0, Tq]`` for a single sequence.
    """
    from .sparse_fwd_kvouter_combine import (
        merge_kv_partials,
    )

    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    o_partial, m_partial, l_partial, inv = indexed_block_partials(
        q,
        k_cache,
        v_cache,
        topk_slot_ids,
        kv_to_q_offsets,
        kv_to_q_indices_and_ranks,
        topk=topk,
        block_size=block_size,
        page_size=page_size,
        softmax_scale=softmax_scale,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q,
        used_kv_lens=used_kv_lens,
        q_load_stage=q_load_stage,
        partial_dtype=partial_dtype,
        inv=inv,
        num_splits=num_splits,
        sel_slots=sel_slots,
        sel_offsets=sel_offsets,
        num_sel=num_sel,
    )
    return merge_kv_partials(
        o_partial, m_partial, l_partial, inv, out_dtype=out_dtype, return_lse=return_lse, out=out
    )
