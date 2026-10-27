# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# FlashAttentionForwardCombine is derived from FlashAttention-4's Cute-DSL
# reimplementation of the CUTLASS forward-combine kernel:
# https://github.com/Dao-AILab/flash-attention/blob/6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15/flash_attn/cute/flash_fwd_combine.py
#
# The combine kernel remains under the upstream BSD-3-Clause terms. Fireworks
# authored the KV-outer host-side compile/launch harness and integration changes.
# Copyright (c) 2026 Fireworks AI
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
#
# This module imports ``cutlass`` at top level and is imported lazily by callers.
import math
import os
from functools import partial
from typing import Type, Optional, Tuple

import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, Boolean, const_expr

from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor
from flash_attn.cute.seqlen_info import SeqlenInfo
from cutlass.cute import FastDivmodDivisor

__all__ = ["merge_kv_partials", "FlashAttentionForwardCombine"]


class FlashAttentionForwardCombine:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 8,
        k_block_size: int = 64,
        log_max_splits: int = 4,
        num_threads: int = 256,
        stages: int = 4,
    ):
        """
        Forward combine kernel for split attention computation.

        :param dtype: output data type
        :param dtype_partial: partial accumulation data type
        :param head_dim: head dimension
        :param tile_m: m block size
        :param k_block_size: k block size
        :param log_max_splits: log2 of maximum splits
        :param num_threads: number of threads
        :param varlen: whether using variable length sequences
        :param stages: number of pipeline stages
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.max_splits = 1 << log_max_splits
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        num_threads,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [cutlass.Float16, cutlass.BFloat16, Float32]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if tile_m % 8 != 0:
            return False
        max_splits = 1 << log_max_splits
        if max_splits > 256:
            return False
        if (tile_m * max_splits) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # GMEM copy setup for O partial
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = 128 if self.k_block_size % 128 == 0 else (64 if self.k_block_size % 64 == 0 else 32)
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        # Async copy atom for O partial load
        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))  # 4 vals per load
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        # GMEM copy setup for final O (use universal copy for store)
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=async_copy_elems * self.dtype.width,
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tOpartial_layout,
            vOpartial_layout,  # 4 vals per store
        )

        # LSE copy setup with async copy (alignment = 1)
        lse_copy_bits = Float32.width  # 1 element per copy, width is in bits
        m_block_smem = (
            128
            if self.tile_m % 128 == 0
            else (
                64
                if self.tile_m % 64 == 0
                else (32 if self.tile_m % 32 == 0 else (16 if self.tile_m % 16 == 0 else 8))
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        # Async copy atom for LSE load
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(atom_async_copy_lse, tLSE_layout, vLSE_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory
        # ///////////////////////////////////////////////////////////////////////////////

        # Shared memory to register copy for LSE
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  # Must divide warp size

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (self.smem_threads_per_col_lse, self.num_threads // self.smem_threads_per_col_lse),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        # LSE shared memory layout with swizzling to avoid bank conflicts
        # This works for kBlockMSmem = 8, 16, 32, 64, 128, no bank conflicts
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle, 0, cute.make_ordered_layout((8, m_block_smem), order=(1, 0))
        )
        self.smem_layout_lse = cute.tile_to_shape(smem_layout_atom_lse, (self.max_splits, self.tile_m), (0, 1))

        # O partial shared memory layout (simple layout for pipeline stages)
        self.smem_layout_o = cute.make_ordered_layout((self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2))

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mL_partial: Optional[cute.Tensor] = None,
        mInv: Optional[cute.Tensor] = None,
        mLSE: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        varlen_batch_idx: Optional[cute.Tensor] = None,
        semaphore_to_reset: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Type checking
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mL_partial is not None and mL_partial.element_type not in [Float32]):
            raise TypeError("L partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")

        # FLAT (tile-ordered) mode -- mInv is not None: O_partial is [Hkv*Tq*topK*qhead, D]
        # by pair position p, m~/l are [Hkv, Tq*topK*qhead], and inv[hkv, q, rank] -> p (-1 =
        # never materialized). The split dimension is resolved per (row, split) through inv.
        flat_mode = const_expr(mInv is not None)
        if const_expr(flat_mode):
            if const_expr(len(mO_partial.shape) != 2 or len(mLSE_partial.shape) != 1):
                raise ValueError("flat mode wants O_partial [R, D] and stats flat [Hkv*S]")
            if const_expr(mL_partial is None):
                raise ValueError("flat mode requires (m~, l) stats")
            if const_expr(cu_seqlens is not None or seqused is not None):
                raise ValueError("flat mode does not support varlen")
        # Shape validation - input tensors are in user format, need to be converted to kernel format
        if const_expr(not flat_mode and len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(not flat_mode and len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError("LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)")

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        if const_expr(not flat_mode):
            # (num_splits, b, seqlen, h, d) -> (seqlen, d, num_splits, h, b)
            # or (num_splits, total_q, h, d) -> (total_q, d, num_splits, h)
            O_partial_layout_transpose = [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
            mO_partial = cute.make_tensor(
                mO_partial.iterator, cute.select(mO_partial.layout, mode=O_partial_layout_transpose)
            )
        # (b, seqlen, h, d) -> (seqlen, d, h, b) or (total_q, h, d) -> (total_q, d, h)
        O_layout_transpose = [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        if const_expr(not flat_mode):
            # (num_splits, b, seqlen, h) -> (seqlen, num_splits, h, b)
            # or (num_splits, total_q, h) -> (total_q, num_splits, h)
            LSE_partial_layout_transpose = [2, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
            mLSE_partial = cute.make_tensor(
                mLSE_partial.iterator,
                cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
            )
            # Deferred-normalization mode: mLSE_partial carries m~ (exp2-space row max) and
            # mL_partial carries l (row sum of exp2); O_partial is the RAW (unnormalized)
            # accumulator. Same layout/partitioning as the LSE plane.
            if const_expr(mL_partial is not None):
                mL_partial = cute.make_tensor(
                    mL_partial.iterator,
                    cute.select(mL_partial.layout, mode=LSE_partial_layout_transpose),
                )
        # (b, h, seqlen) -> (seqlen, h, b) or (total_q, h) -> (total_q, h)
        # Non-varlen output LSE is allocated head-major (b, Hq, Tq) so the kernel writes the
        # final LSE as [Hq, Tq] directly (matching the FlashAttention forward convention) —
        # no host-side transpose. The store is 1 element per copy, so the gmem layout is free.
        LSE_layout_transpose = [2, 1, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )

        # Determine if we have variable length sequences
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()

        if const_expr(mL_partial is not None):

            @cute.struct
            class SharedStorage:
                sLSE: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128]
                sL: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128]
                sMaxValidSplit: cute.struct.Align[cute.struct.MemRange[Int32, self.tile_m], 128]
                sO: cute.struct.Align[cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128]

        else:

            @cute.struct
            class SharedStorage:
                sLSE: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128]
                sMaxValidSplit: cute.struct.Align[cute.struct.MemRange[Int32, self.tile_m], 128]
                sO: cute.struct.Align[cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128]

        smem_size = SharedStorage.size_in_bytes()

        # Grid dimensions: (ceil_div(seqlen, m_block), ceil_div(head_dim, k_block), num_head * batch)
        if const_expr(flat_mode):
            seqlen = mO.shape[0]   # output O is (seqlen, d, h, b) post-transpose
            num_head = mO.shape[2]
            batch_size = mO.shape[3]
        else:
            seqlen = mO_partial.shape[0]
            num_head = mO_partial.shape[3]
            batch_size = mO_partial.shape[4] if const_expr(cu_seqlens is None) else Int32(cu_seqlens.shape[0] - 1)

        # Create FastDivmodDivisor objects for efficient division
        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        grid_dim = (
            cute.ceil_div(seqlen * num_head, self.tile_m),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch_size,
        )

        self.kernel(
            mO_partial,
            mLSE_partial,
            mO,
            mL_partial,
            mInv,
            mLSE,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            varlen_batch_idx,
            semaphore_to_reset,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            seqlen_divmod,
            head_divmod,
            varlen,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mL_partial: Optional[cute.Tensor],
        mInv: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        varlen: cutlass.Constexpr[bool],
    ):
        # Thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()

        # Map virtual batch index to real batch index (for persistent tile schedulers)
        batch_idx = (
            varlen_batch_idx[maybe_virtual_batch] if const_expr(varlen_batch_idx is not None) else maybe_virtual_batch
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sL = storage.sL.get_tensor(smem_layout_lse) if const_expr(mL_partial is not None) else None
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)

        # Handle semaphore reset — wait for dependent grids first
        if const_expr(semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and maybe_virtual_batch == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0

        flat_mode = const_expr(mInv is not None)
        # Get number of splits (use maybe_virtual_batch for per-batch-slot splits)
        if const_expr(flat_mode):
            num_splits = Int32(mInv.shape[2])
        else:
            num_splits = (
                num_splits_dynamic_ptr[maybe_virtual_batch]
                if const_expr(num_splits_dynamic_ptr is not None)
                else mLSE_partial.shape[1]
            )
        # Handle variable length sequences using SeqlenInfo
        seqlen_info = SeqlenInfo.create(
            batch_idx=batch_idx,
            seqlen_static=mInv.shape[1] if const_expr(flat_mode) else mO_partial.shape[0],
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            # Don't need to pass in tile size since we won't use offset_padded
        )
        seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

        # Extract number of heads (head index will be determined dynamically)
        num_head = mO.shape[2] if const_expr(flat_mode) else mO_partial.shape[3]
        max_idx = seqlen * num_head
        if const_expr(flat_mode):
            # Flat addressing helpers: inv [Hkv, Tq, topK]; stats [Hkv, S]; O [R, D].
            # qhead = Hq / Hkv; pair segment G = Tq*topK; row(hkv, p, hl) = (hkv*G + p)*qhead + hl.
            n_hkv_f = Int32(mInv.shape[0])
            qhead_f = Int32(num_head) // n_hkv_f
            pairs_g = Int32(mInv.shape[1]) * Int32(mInv.shape[2])

        # Early exit for single split if dynamic
        if (const_expr(num_splits_dynamic_ptr is None) or num_splits > 1) and (
            const_expr(not varlen) or m_block * self.tile_m < max_idx
        ):
            # Wait for dependent grids (e.g., the main attention kernel that produces O_partial/LSE_partial)
            cute.arch.griddepcontrol_wait()

            # ===============================
            # Step 1: Load LSE_partial from gmem to shared memory
            # ===============================

            gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
            tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
            if const_expr(not flat_mode):
                mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
                mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))
            else:
                # Flat stats planes passed as 1D [Hkv*S]: 1-element tiles; per (row, split)
                # the element index is (hkv*G + p)*qhead + hl.
                mLSE_flat_copy = cute.tiled_divide(mLSE_partial, (1,))
            if const_expr(mL_partial is not None):
                tLsL = gmem_thr_copy_LSE.partition_D(sL)
                if const_expr(not flat_mode):
                    mL_partial_cur = seqlen_info.offset_batch(mL_partial, batch_idx, dim=3)
                    mL_partial_copy = cute.tiled_divide(mL_partial_cur, (1,))
                else:
                    mL_flat_copy = cute.tiled_divide(mL_partial, (1,))
            # Create identity tensor for coordinate tracking
            cLSE = cute.make_identity_tensor((self.max_splits, self.tile_m))
            tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)

            # Load LSE partial values
            for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                mi = tLSEcLSE[0, 0, m][1]  # Get m coordinate
                idx = m_block * self.tile_m + mi
                if idx < max_idx:
                    # Calculate actual sequence position and head using FastDivmodDivisor.
                    # FLAT mode rows are TOKEN-MAJOR (idx = q*num_head + head): one query's
                    # qhead head-rows are adjacent in the tile, so their flat-layout gathers
                    # (p*qhead + hl) hit CONTIGUOUS gmem rows (sec 7w).
                    if const_expr(flat_mode):
                        m_idx, head_idx = divmod(idx, head_divmod)
                    elif const_expr(not varlen):
                        head_idx, m_idx = divmod(idx, seqlen_divmod)
                    else:
                        head_idx = idx // seqlen
                        m_idx = idx - head_idx * seqlen
                    if const_expr(not flat_mode):
                        mLSE_partial_cur_copy = mLSE_partial_copy[None, m_idx, None, head_idx]
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            si = tLSEcLSE[0, s, 0][0]  # Get split coordinate
                            if si < num_splits:
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_partial_cur_copy[None, si],
                                    tLSEsLSE[None, s, m],
                                )
                            else:
                                tLSEsLSE[None, s, m].fill(-Float32.inf)
                        if const_expr(mL_partial is not None):
                            mL_partial_cur_copy = mL_partial_copy[None, m_idx, None, head_idx]
                            for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                                si = tLSEcLSE[0, s, 0][0]
                                if si < num_splits:
                                    cute.copy(
                                        gmem_thr_copy_LSE,
                                        mL_partial_cur_copy[None, si],
                                        tLsL[None, s, m],
                                    )
                                else:
                                    tLsL[None, s, m].fill(0.0)
                    else:
                        # FLAT mode: resolve (q=m_idx, split) -> pair position p via the
                        # inverse map; p < 0 (never materialized) -> (-inf, 0) so the
                        # weight is exactly zero. Element = stats[hkv, p*qhead + hl].
                        hkv_m = head_idx // qhead_f
                        hl_m = head_idx - hkv_m * qhead_f
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            si = tLSEcLSE[0, s, 0][0]
                            pi = Int32(-1)
                            if si < num_splits:
                                pi = Int32(mInv[hkv_m, m_idx, si])
                            if pi >= 0:
                                elem = (hkv_m * pairs_g + pi) * qhead_f + hl_m
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_flat_copy[None, elem],
                                    tLSEsLSE[None, s, m],
                                )
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mL_flat_copy[None, elem],
                                    tLsL[None, s, m],
                                )
                            else:
                                tLSEsLSE[None, s, m].fill(-Float32.inf)
                                tLsL[None, s, m].fill(0.0)
                # Don't need to zero out the rest of the LSEs, as we will not write the output to gmem
            cute.arch.cp_async_commit_group()

            # ===============================
            # Step 2: Load O_partial for pipeline stages
            # ===============================

            gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
            cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
            tOcO = gmem_thr_copy_O_partial.partition_D(cO)
            tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)
            mO_partial_cur = mO_partial if const_expr(flat_mode) else seqlen_info.offset_batch(mO_partial, batch_idx, dim=4)

            # Precompute these values to avoid recomputing them in the loop
            num_rows = const_expr(cute.size(tOcO, mode=[1]))
            tOmidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOhidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOrOptr = cute.make_rmem_tensor(num_rows, cutlass.Int64)
            tOrInvPtr = cute.make_rmem_tensor(num_rows, cutlass.Int64) if const_expr(flat_mode) else None
            for m in cutlass.range(num_rows, unroll_full=True):
                mi = tOcO[0, m, 0][0]  # m coordinate
                idx = m_block * self.tile_m + mi
                if const_expr(flat_mode):
                    tOmidx[m], tOhidx[m] = divmod(idx, head_divmod)  # token-major (sec 7w)
                elif const_expr(not varlen):
                    tOhidx[m], tOmidx[m] = divmod(idx, seqlen_divmod)
                else:
                    tOhidx[m] = idx // seqlen
                    tOmidx[m] = idx - tOhidx[m] * seqlen
                if const_expr(not flat_mode):
                    tOrOptr[m] = utils.elem_pointer(
                        mO_partial_cur, (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m])
                    ).toint()
                else:
                    # FLAT mode: base pointer at the row's segment origin (p = 0); the O
                    # loader offsets by p*qhead rows after reading p from the inverse map.
                    hkv_m2 = tOhidx[m] // qhead_f
                    hl_m2 = tOhidx[m] - hkv_m2 * qhead_f
                    row0 = hkv_m2 * pairs_g * qhead_f + hl_m2
                    tOrOptr[m] = utils.elem_pointer(
                        mO_partial_cur, (row0, k_block * self.k_block_size)
                    ).toint()
                    tOrInvPtr[m] = utils.elem_pointer(mInv, (hkv_m2, tOmidx[m], 0)).toint()
                if idx >= max_idx:
                    tOhidx[m] = -1

            tOpO = None
            if const_expr(not self.is_even_k):
                tOpO = cute.make_rmem_tensor(cute.size(tOcO, mode=[2]), Boolean)
                for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                    tOpO[k] = tOcO[0, 0, k][1] < mO_partial.shape[1] - k_block * self.k_block_size

            load_O_partial = partial(
                self.load_O_partial,
                gmem_tiled_copy_O_partial,
                tOrOptr,
                tOsO_partial,
                tOhidx,
                tOpO,
                tOcO,
                mO_partial_cur.layout,
                tOrInvPtr,
                qhead_f if const_expr(flat_mode) else Int32(0),
            )

            # Load first few stages of O_partial
            for stage in cutlass.range(self.stages - 1, unroll_full=True):
                if stage < num_splits:
                    load_O_partial(stage, stage)
                cute.arch.cp_async_commit_group()

            # ===============================
            # Step 3: Load and transpose LSE from smem to registers
            # ===============================

            # Wait for LSE and initial O partial stages to complete
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()

            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
            ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
            ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
            cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)
            if const_expr(mL_partial is not None):
                ts2rsL = s2r_thr_copy_LSE.partition_S(sL)
                ts2rrL = cute.make_rmem_tensor_like(ts2rsL)
                cute.copy(s2r_tiled_copy_LSE, ts2rsL, ts2rrL)

            # ===============================
            # Step 4: Compute final LSE along split dimension
            # ===============================

            if const_expr(mLSE is not None):
                lse_sum = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
            ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
            # We compute the max valid split for each row to short-circuit the computation later
            max_valid_split = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Int32)
            assert cute.size(ts2rrLSE, mode=[0]) == 1
            # Compute max, scales, and final LSE for each row
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                # Find max LSE value across splits
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                lse_max = cute.arch.warp_reduction_max(
                    ts2rrLSE[None, None, m]
                    .load()
                    .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                    threads_in_group=threads_per_col,
                )
                # Find max valid split index
                max_valid_idx = -1
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    if ts2rrLSE[0, s, m] != -Float32.inf:
                        max_valid_idx = ts2rcLSE[0, s, 0][0]  # Get split coordinate
                max_valid_split[m] = cute.arch.warp_reduction_max(max_valid_idx, threads_in_group=threads_per_col)
                # Compute exp scales and sum
                lse_max_cur = 0.0 if lse_max == -Float32.inf else lse_max  # In case all local LSEs are -inf
                LOG2_E = math.log2(math.e)
                LN_2 = math.log(2.0)
                lse_sum_cur = 0.0
                if const_expr(mL_partial is not None):
                    # Deferred-normalization mode: the "LSE" plane holds m~ (exp2-space row
                    # max) and O_partial is RAW. Per split: a_s = exp2(m~_s - M~); the merge
                    # denominator is D = sum_s a_s * l_s (so the per-split O weight a_s / D
                    # both combines and normalizes); final LSE = ln(D) + M~ * ln2.
                    for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                        a = cute.math.exp2(ts2rrLSE[0, s, m] - lse_max_cur, fastmath=True)
                        lse_sum_cur += a * ts2rrL[0, s, m]
                        ts2rrLSE[0, s, m] = a  # Store weight numerator for later use
                else:
                    for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                        scale = cute.math.exp2(ts2rrLSE[0, s, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True)
                        lse_sum_cur += scale
                        ts2rrLSE[0, s, m] = scale  # Store scale for later use
                lse_sum_cur = cute.arch.warp_reduction_sum(lse_sum_cur, threads_in_group=threads_per_col)
                if const_expr(mLSE is not None):
                    if const_expr(mL_partial is not None):
                        lse_sum[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max * LN_2
                    else:
                        lse_sum[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                # Normalize scales
                inv_sum = 0.0 if (lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur) else 1.0 / lse_sum_cur
                ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
            # Store the scales exp(lse - lse_logsum) back to smem
            cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

            # Store max valid split to smem
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                    mi = ts2rcLSE[0, 0, m][1]
                    if mi < self.tile_m:
                        sMaxValidSplit[mi] = max_valid_split[m]

            # ===============================
            # Step 5: Store final LSE to gmem
            # ===============================

            if const_expr(mLSE is not None):
                if const_expr(cu_seqlens is None):
                    mLSE_cur = mLSE[None, None, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset((offset, 0), mLSE)
                if k_block == 0:  # Only first k_block writes LSE when mLSE is provided
                    for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                        if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                            mi = ts2rcLSE[0, 0, m][1]
                            idx = m_block * self.tile_m + mi
                            if idx < max_idx:
                                if const_expr(flat_mode):
                                    m_idx, head_idx = divmod(idx, head_divmod)  # token-major
                                elif const_expr(not varlen):
                                    head_idx, m_idx = divmod(idx, seqlen_divmod)
                                else:
                                    head_idx = idx // seqlen
                                    m_idx = idx - head_idx * seqlen
                                mLSE_cur[m_idx, head_idx] = lse_sum[m]

            # ===============================
            # Step 6: Read O_partial and accumulate final O
            # ===============================

            cute.arch.sync_threads()

            # Get max valid split for this thread
            thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
            for m in cutlass.range(1, cute.size(tOcO, mode=[1]), unroll_full=True):
                thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[tOcO[0, m, 0][0]])

            tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
            tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
            tOrO.fill(0.0)

            stage_load = self.stages - 1
            stage_compute = 0

            # Main accumulation loop
            for s in cutlass.range(thr_max_valid_split + 1, unroll=4):
                # Get scales for this split
                scale = cute.make_rmem_tensor(num_rows, Float32)
                for m in cutlass.range(num_rows, unroll_full=True):
                    scale[m] = sLSE[s, tOcO[0, m, 0][0]]  # Get scale from smem

                # Load next stage if needed
                split_to_load = s + self.stages - 1
                if split_to_load <= thr_max_valid_split:
                    load_O_partial(split_to_load, stage_load)
                cute.arch.cp_async_commit_group()
                stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

                # Wait for the current stage to be ready
                cute.arch.cp_async_wait_group(self.stages - 1)
                # We don't need __syncthreads() because each thread is just reading its own data from smem
                # Copy from smem to registers
                cute.autovec_copy(tOsO_partial[None, None, None, stage_compute], tOrO_partial)
                stage_compute = 0 if stage_compute == self.stages - 1 else stage_compute + 1

                # Accumulate scaled partial results
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0 and scale[m] > 0.0:
                        tOrO[None, m, None].store(
                            tOrO[None, m, None].load() + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                        )

            # ===============================
            # Step 7: Write final O to gmem
            # ===============================

            rO = cute.make_rmem_tensor_like(tOrO, self.dtype)
            rO.store(tOrO.load().to(self.dtype))
            mO_cur = seqlen_info.offset_batch(mO, batch_idx, dim=3)
            if const_expr(cu_seqlens is None):
                mO_cur = mO[None, None, None, batch_idx]
            else:
                mO_cur = cute.domain_offset((offset, 0, 0), mO)
            mO_cur = utils.domain_offset_aligned((0, k_block * self.k_block_size, 0), mO_cur)
            elems_per_store = const_expr(cute.size(gmem_tiled_copy_O.layout_tv_tiled[1]))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # Write final results
            for m in cutlass.range(num_rows, unroll_full=True):
                if tOhidx[m] >= 0:
                    mO_cur_copy = cute.tiled_divide(mO_cur[tOmidx[m], None, tOhidx[m]], (elems_per_store,))
                    for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                        k_idx = tOcO[0, 0, k][1] // elems_per_store
                        if const_expr(self.is_even_k) or tOpO[k]:
                            cute.copy(gmem_thr_copy_O, rO[None, m, k], mO_cur_copy[None, k_idx])

    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOpO: Optional[cute.Tensor],
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        tOrInvPtr: Optional[cute.Tensor],
        qhead_f: Int32,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1]))
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        flat = const_expr(tOrInvPtr is not None)
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                if const_expr(not flat):
                    o_gmem_ptr = cute.make_ptr(
                        tOsO_partial.element_type, tOrOptr[m], cute.AddressSpace.gmem, assumed_align=16
                    )
                    mO_partial_cur = cute.make_tensor(
                        o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                    )
                    mO_partial_cur_copy = cute.tiled_divide(mO_partial_cur, (elems_per_load,))
                    for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                        k_idx = tOcO[0, 0, k][1] // elems_per_load
                        if const_expr(tOpO is None) or tOpO[k]:
                            cute.copy(
                                gmem_tiled_copy_O_partial,
                                mO_partial_cur_copy[None, k_idx, split],
                                tOsO_partial_cur[None, m, k],
                            )
                else:
                    # FLAT (tile-ordered) mode: p = inv[row, split]; the row's O data lives
                    # at segment base + p*qhead rows. p < 0 (never materialized) -> zero-fill
                    # the smem chunk so the (zero-weighted) accumulation stays NaN-free.
                    inv_ptr = cute.make_ptr(
                        cutlass.Int32, tOrInvPtr[m], cute.AddressSpace.gmem, assumed_align=4
                    )
                    p = Int32(cute.make_tensor(inv_ptr, (self.max_splits,))[split])
                    row_elems = cute.size(mO_cur_partial_layout, mode=[1])
                    byte_off = cutlass.Int64(p) * qhead_f * row_elems * const_expr(
                        tOsO_partial.element_type.width // 8
                    )
                    o_gmem_ptr = cute.make_ptr(
                        tOsO_partial.element_type,
                        tOrOptr[m] + byte_off,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    mO_partial_cur = cute.make_tensor(o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None)))
                    mO_partial_cur_copy = cute.tiled_divide(mO_partial_cur, (elems_per_load,))
                    for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                        k_idx = tOcO[0, 0, k][1] // elems_per_load
                        if const_expr(tOpO is None) or tOpO[k]:
                            if p >= 0:
                                cute.copy(
                                    gmem_tiled_copy_O_partial,
                                    mO_partial_cur_copy[None, k_idx],
                                    tOsO_partial_cur[None, m, k],
                                )
                            else:
                                tOsO_partial_cur[None, m, k].fill(0)


# --------------------------------------------------------------------------- #
# Host-side compile/launch harness (bespoke; mirrors this package's launch
# conventions). Compiles the vendored kernel above via to_cute_tensor templates.
# --------------------------------------------------------------------------- #
_compile_cache: dict = {}

_TORCH2CUTE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}


def _get_compiled(out_dtype, partial_dtype, head_dim, log_max_splits, has_lse, has_l, has_inv, templates):
    key = (out_dtype, partial_dtype, head_dim, log_max_splits, has_lse, has_l, has_inv)
    if key not in _compile_cache:
        num_threads = 128
        # Same heuristics as FA4's _flash_attn_fwd_combine for head_dim > 64.
        k_block_size = 64 if head_dim <= 64 else 128
        k_block_gmem = 128 if k_block_size % 128 == 0 else (64 if k_block_size % 64 == 0 else 32)
        # The partial load/store uses ONE thread tiling that must cover exactly
        # (tile_m, k_block_gmem): its thread-row count is
        # num_threads // (k_block_gmem // async_copy_elems), which must equal tile_m or the
        # grid over-spans the tile (OOB). async_copy_elems = 128 / partial_width, so for
        # bf16/fp16 partials it doubles vs fp32 and tile_m must double too. For fp32 this
        # reduces to FA4's original tile_m (8/16/32).
        async_copy_elems = 128 // _TORCH2CUTE[partial_dtype].width
        tile_m = num_threads * async_copy_elems // k_block_gmem
        kernel = FlashAttentionForwardCombine(
            dtype=_TORCH2CUTE[out_dtype],
            dtype_partial=_TORCH2CUTE[partial_dtype],
            head_dim=head_dim,
            tile_m=tile_m,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
            num_threads=num_threads,
        )
        op_t, lp_t, l_t, inv_t, o_t, lse_t, cu_t = templates
        _compile_cache[key] = cute.compile(
            kernel,
            op_t,
            lp_t,
            o_t,
            l_t if has_l else None,
            inv_t if has_inv else None,
            lse_t if has_lse else None,
            cu_t,
            None,
            None,
            None,
            None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _compile_cache[key]


def merge_kv_partials(
    o_partial: torch.Tensor,  # legacy [Hq, Tq, topK, D] | flat [Hkv*Tq*topK*qhead, D]
    lse_partial: torch.Tensor,  # legacy LSE / m~ [Hq, Tq, topK] | flat m~ [Hkv, Tq*topK*qhead]
    l_partial: Optional[torch.Tensor] = None,  # row sums, same shape as lse_partial
    inv: Optional[torch.Tensor] = None,  # [Hkv, Tq, topK] int32 (q, rank) -> pair pos (flat mode)
    out_dtype: torch.dtype = torch.bfloat16,
    return_lse: bool = True,
    out: Optional[torch.Tensor] = None,  # preallocated [Tq, Hq, D] out_dtype (written in place)
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge the ``topK`` per-(q, rank) partials into the final output.

    Two modes:
    * legacy: ``o_partial`` is NORMALIZED and ``lse_partial`` is the per-rank LSE
      (unused ranks carry ``-inf``).
    * deferred normalization (``l_partial`` given): ``o_partial`` is the RAW exp2
      accumulator, ``lse_partial`` carries ``m~ = row_max * scale_log2`` (exp2-space)
      and ``l_partial`` the row sums; the combine computes the per-rank weights
      ``exp2(m~_i - M~) / sum_j exp2(m~_j - M~) * l_j`` (which normalizes AND merges)
      and the final LSE ``ln(D) + M~*ln2``. Unused ranks carry ``(m~=-inf, l=0)``.

    Uses the vendored combine kernel above. The split dimension is ``topK``
    (``num_splits``). ``o_partial`` may be fp32 or low precision (bf16/fp16) — the combine
    loads it as ``dtype_partial`` and accumulates the lse-weighted sum in fp32, so
    bf16 partials halve the partial read/write traffic at the cost of the partials'
    own rounding. Returns ``(o [Tq, Hq, D] out_dtype, lse [Hq, Tq] fp32 or None)`` — the LSE
    is head-major (the FlashAttention forward convention), written directly by the kernel.
    """
    flat_mode = inv is not None
    if flat_mode:
        # TILE-ORDERED flat partials (sec 7t): shapes are fixed across requests; the kernel
        # resolves (q, rank) -> pair position through ``inv`` (-1 = never materialized).
        assert o_partial.dim() == 2 and lse_partial.dim() == 2 and l_partial is not None
        hkv_n, tq, topk = inv.shape
        r_total, d = o_partial.shape
        qhead = r_total // (hkv_n * tq * topk)
        assert r_total == hkv_n * tq * topk * qhead
        hq = hkv_n * qhead
        assert lse_partial.shape == (hkv_n, tq * topk * qhead) == l_partial.shape
        assert inv.dtype == torch.int32 and lse_partial.dtype == torch.float32
    else:
        assert o_partial.dim() == 4, f"o_partial must be [Hq, Tq, topK, D], got {tuple(o_partial.shape)}"
        assert lse_partial.dim() == 3, f"lse_partial must be [Hq, Tq, topK], got {tuple(lse_partial.shape)}"
        hq, tq, topk, d = o_partial.shape
        assert lse_partial.shape == (hq, tq, topk)
        assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"
        if l_partial is not None:
            assert l_partial.shape == (hq, tq, topk) and l_partial.dtype == torch.float32
    assert o_partial.dtype in _TORCH2CUTE, f"unsupported o_partial dtype {o_partial.dtype}"
    partial_dtype = o_partial.dtype

    device = o_partial.device
    # Combine's non-varlen path wants a batch dimension. Use a metadata-only
    # B=1 view instead of allocating a tiny cu_seqlens tensor every call.
    # It wants out_partial (num_splits, batch, total_q, nheads, d) with d
    # contiguous, and lse_partial (num_splits, batch, total_q, nheads).
    # Strided view (no 8.6GB copy): D stays contiguous (stride 1), which is all the
    # combine kernel's cp.async loads require.
    if flat_mode:
        op = o_partial  # [R, D] flat, consumed via inv
        lp = lse_partial.reshape(-1)  # [Hkv*S] 1D flat
        ll = l_partial.reshape(-1)
    else:
        op = o_partial.permute(2, 1, 0, 3).unsqueeze(1)  # (topK, 1, Tq, Hq, D), D contiguous
        lp = lse_partial.permute(2, 1, 0).unsqueeze(1)  # (topK, 1, Tq, Hq), topK stride-1 after transpose
        ll = l_partial.permute(2, 1, 0).unsqueeze(1) if l_partial is not None else None
    if out is None:
        out = torch.empty(tq, hq, d, dtype=out_dtype, device=device)  # (Tq, Hq, D)
    else:
        assert out.shape == (tq, hq, d), f"out shape {tuple(out.shape)} != {(tq, hq, d)}"
        assert out.dtype == out_dtype and out.device == device and out.is_contiguous(), (
            "out must be a contiguous [Tq, Hq, D] tensor of out_dtype on q.device"
        )
    out_batched = out.unsqueeze(0)
    # LSE is allocated head-major (batch, Hq, Tq) with Tq contiguous; combined with the kernel's
    # LSE_layout_transpose this makes the kernel write the final LSE as [Hq, Tq] directly (no host
    # transpose), matching the FlashAttention forward convention.
    lse = torch.empty(1, hq, tq, dtype=torch.float32, device=device) if return_lse else None

    op_t = to_cute_tensor(op, assumed_align=16, leading_dim=(1 if flat_mode else 4))
    lp_t = to_cute_tensor(lp, assumed_align=4, leading_dim=0)
    l_t = to_cute_tensor(ll, assumed_align=4, leading_dim=0) if ll is not None else None
    inv_t = to_cute_tensor(inv, assumed_align=4, leading_dim=2) if inv is not None else None
    o_t = to_cute_tensor(out_batched, assumed_align=16, leading_dim=3)
    lse_t = to_cute_tensor(lse, assumed_align=4, leading_dim=2) if return_lse else None

    log_max_splits = max(math.ceil(math.log2(max(topk, 2))), 5)
    compiled = _get_compiled(
        out_dtype,
        partial_dtype,
        d,
        log_max_splits,
        return_lse,
        ll is not None,
        inv is not None,
        (op_t, lp_t, l_t, inv_t, o_t, lse_t, None),
    )
    compiled(op, lp, out_batched, ll, inv, lse, None, None, None, None, None)

    return out, lse.squeeze(0) if lse is not None else None
