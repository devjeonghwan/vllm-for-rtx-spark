# Copyright (c) 2026 Fireworks AI
# SPDX-License-Identifier: Apache-2.0

"""Ahead-of-time (AOT) export of the KV-outer sparse-attention cutedsl kernels.

The Python path JIT-compiles each kernel once per a small *compile key* and
then runs it for any batch/seqlen (shape scalars are runtime args). For the C++
backend we instead compile each kernel through the **CuTe ABI** (no tvm-ffi,
explicit ``cuda.CUstream`` arg) and ``dump_to_object`` it to a ``.o`` that the
C++ op loads with ``CuteDSLRT_Module_Create_From_Bytes`` (see
``csrc/m3_sparse_attention``).

This module is the single source of truth for:

* :func:`compile_keys_for_request` -- the exact compile key of every kernel for a
  request (so the C++ op can strict-check that no request would require a
  recompile -- a fatal error, never a silent JIT).
* :func:`export_all_reachable` -- compile + ``dump_to_object`` every kernel in the
  full reachable key set for a deployment config (both offsets variants, etc.),
  returning per-kernel ``.o`` paths plus the C-ABI arg descriptors the C++ op
  needs to pack arguments.

``cute.compile`` is only ever invoked from here (guarded by :data:`_AOT_ALLOWED`),
so the hot path can never trigger a recompile.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from .build_kvouter_index import (
    _adaptive_replicas,
    _CountEdgesKernel,
    _CountToOffsetsParallelKernel,
    _COUNT_REPLICAS_FLOOR,
    _COUNT_REPLICAS_MAX,
    _COUNT_REPLICAS_OVERRIDE,
    _COUNT_TO_OFFSETS_PARALLEL_THRESHOLD,
    _InitSlotsAndCountsKernel,
    _ReduceReplicasKernel,
    _ScatterRanksKernel,
    _make_count_to_offsets_kernel,
)


def _replica_values() -> list[int]:
    """All replica counts `_adaptive_replicas` can return, so every reachable value is
    pre-exported and the request-variable replica choice never triggers a recompile (or,
    in the C++ backend, a fatal missing-kernel error). When MINIMAX_KERNELS_KVOUTER_COUNT_REPLICAS is
    set, `_adaptive_replicas` returns exactly that (possibly non-power-of-two) value, so
    that single value is the only reachable one; otherwise it is the powers of two in
    [floor, max]."""
    if _COUNT_REPLICAS_OVERRIDE is not None:
        return [max(1, int(_COUNT_REPLICAS_OVERRIDE))]
    vals, r = [], _COUNT_REPLICAS_FLOOR
    while r <= _COUNT_REPLICAS_MAX:
        vals.append(r)
        r <<= 1
    return vals


from .sparse_fwd_kvouter import (
    SparseKVOuterForward,
    _arch_defaults,
)
from .sparse_fwd_kvouter_load_balance_schedule import (
    _LoadBalanceScheduler,
)

__all__ = [
    "AotConfig",
    "ArgDesc",
    "KernelArtifact",
    "compile_keys_for_request",
    "export_all_reachable",
]

# Guard: cute.compile (recompilation) is only allowed while exporting. Any attempt
# to compile outside an export call is a bug in the no-recompile contract.
_AOT_ALLOWED = False

_SCALAR_CTYPE = {Int32: "int32", Int64: "int64", Float32: "float32"}

_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
}
_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AotConfig:
    """Everything fixed for a deployment (the cross product of compile keys is
    derived from this). Per-request dynamic values (Tq, B, cu_seqlens, ...) are
    NOT here -- they are runtime kernel args."""

    nheads_kv: int
    qhead: int  # Hq // Hkv
    topk: int
    block_size: int
    page_size: int
    causal: bool
    head_dim: int
    num_splits: int  # device SM count
    q_dtype: str  # one of _TORCH_DTYPE keys
    out_dtype: str
    partial_dtype: str
    store_in_corr: bool
    has_block_tables: bool = True
    return_lse: bool = False

    @property
    def ratio(self) -> int:
        return self.block_size // self.page_size

    @property
    def hkv(self) -> int:
        return self.nheads_kv

    def q_torch(self) -> torch.dtype:
        return _TORCH_DTYPE[self.q_dtype]

    def out_torch(self) -> torch.dtype:
        return _TORCH_DTYPE[self.out_dtype]

    def partial_torch(self) -> torch.dtype:
        return _TORCH_DTYPE[self.partial_dtype]


# --------------------------------------------------------------------------- #
# Arg descriptors (the C-ABI the C++ op uses to pack arguments)
# --------------------------------------------------------------------------- #
@dataclass
class ArgDesc:
    """C-ABI descriptor for one exported-kernel argument (consumed by the C++ packer).

    Attributes:
        name: The argument's name in the kernel's ``__call__`` signature.
        kind: One of ``"tensor"`` | ``"scalar"`` | ``"stream"``.
        rank: (tensor) Number of dimensions.
        dynamic_shapes_mask: (tensor) Per-dim 1/0 — which shape dims are dynamic ABI fields.
        dynamic_strides_mask: (tensor) Per-dim 1/0 — which strides are dynamic ABI fields
            (the contiguous leading dim has a static stride and is 0).
        use_32bit_stride: (tensor) True if dynamic strides are int32, else int64.
        scalar_dtype: (scalar) One of ``"int32"`` | ``"int64"`` | ``"float32"``.
    """

    name: str
    kind: str  # "tensor" | "scalar" | "stream"
    rank: Optional[int] = None
    dynamic_shapes_mask: Optional[list[int]] = None
    dynamic_strides_mask: Optional[list[int]] = None
    use_32bit_stride: Optional[bool] = None
    scalar_dtype: Optional[str] = None  # "int32" | "int64" | "float32"


@dataclass
class KernelArtifact:
    """One AOT-exported kernel: its compile key, the ``.o`` path + symbol, and arg ABI.

    Attributes:
        kernel: Logical kernel name (e.g. ``"forward"``, ``"combine"``).
        key: The compile key (json-serializable) this artifact was built for.
        function_prefix: Symbol prefix the ``.o`` was exported with (CuteDSLRT lookup key).
        object_path: Filesystem path to the exported ``.o``.
        args: Ordered per-argument C-ABI descriptors (the C++ op packs args in this order).
    """

    kernel: str
    key: list  # the compile key (json-serializable)
    function_prefix: str
    object_path: str
    args: list[ArgDesc] = field(default_factory=list)


def _classify(name: str, val: Any) -> Optional[ArgDesc]:
    """Map a compile-template argument to its C-ABI descriptor (None => omitted)."""
    if val is None:
        return None
    if hasattr(val, "dynamic_shapes_mask"):  # cute runtime tensor
        return ArgDesc(
            name=name,
            kind="tensor",
            rank=len(val.shape),
            dynamic_shapes_mask=[int(x) for x in val.dynamic_shapes_mask],
            dynamic_strides_mask=[int(x) for x in val.dynamic_strides_mask],
            use_32bit_stride=bool(val._use_32bit_stride),
        )
    if isinstance(val, cute.runtime._FakeStream):
        return ArgDesc(name=name, kind="stream")
    for cute_t, cname in _SCALAR_CTYPE.items():
        if isinstance(val, cute_t):
            return ArgDesc(name=name, kind="scalar", scalar_dtype=cname)
    raise TypeError(f"cannot classify AOT arg {name!r}: {type(val)}")


# --------------------------------------------------------------------------- #
# Compile keys (verbatim mirror of the JIT cache keys; single source of truth)
# --------------------------------------------------------------------------- #
def _key_init(cfg: AotConfig, replicas: int) -> tuple:
    return (cfg.hkv, cfg.ratio, cfg.page_size, cfg.has_block_tables, replicas)


def _key_count(cfg: AotConfig, replicas: int) -> tuple:
    return (cfg.hkv, cfg.topk, cfg.block_size, cfg.causal, cfg.has_block_tables, cfg.ratio, replicas)


def _key_reduce(replicas: int) -> tuple:
    return (replicas,)


def _key_scatter(cfg: AotConfig, replicas: int) -> tuple:
    return (cfg.hkv, cfg.topk, cfg.block_size, cfg.causal, cfg.has_block_tables, cfg.ratio, replicas)


def _key_offsets(cfg: AotConfig, parallel: bool) -> tuple:
    return (cfg.hkv, bool(parallel))


def _key_scheduler(cfg: AotConfig) -> tuple:
    return (cfg.hkv, cfg.num_splits)


def _qls_obuf(cfg: AotConfig) -> tuple[int, int]:
    return _arch_defaults(cfg.q_torch(), cfg.partial_torch())


def _key_forward(cfg: AotConfig) -> tuple:
    qls, obuf = _qls_obuf(cfg)
    # Mirrors SparseKVOuterForward's JIT key (q_t.element_type, o_t.element_type) where
    # o_t is the flat O_partial buffer -> its element type is PARTIAL dtype, not out_dtype.
    # (The forward never produces the final output; out_dtype only matters in combine.)
    # element_type strings keep the key json-serializable; the C++ side never sees it.
    return (
        cfg.qhead,
        cfg.nheads_kv,
        cfg.page_size,
        cfg.causal,
        qls,
        obuf,
        cfg.q_dtype,
        cfg.partial_dtype,
    )


def _log_max_splits(cfg: AotConfig) -> int:
    import math

    return max(math.ceil(math.log2(max(cfg.topk, 2))), 5)


def _key_combine(cfg: AotConfig) -> tuple:
    # Flat mode: has_l=True, has_inv=True; has_lse=return_lse.
    return (
        cfg.out_dtype,
        cfg.partial_dtype,
        cfg.head_dim,
        _log_max_splits(cfg),
        cfg.return_lse,
        True,
        True,
    )


def parallel_offsets_for_request(num_block_slots: int) -> bool:
    """Whether a request uses the parallel (vs serial) count->offsets kernel.

    Request-variable; both variants are pre-exported, so crossing the threshold
    never triggers a recompile.
    """
    return num_block_slots > _COUNT_TO_OFFSETS_PARALLEL_THRESHOLD


def replicas_for_request(cfg: AotConfig, *, tq: int, num_block_slots: int) -> int:
    """Adaptive replica count for the index-build counters (mirrors
    `_adaptive_replicas`). Request-variable (depends on tq via cap); all reachable
    values (`_replica_values()`) are pre-exported so it never triggers a recompile."""
    cap = tq * cfg.hkv * cfg.topk
    return _adaptive_replicas(cap, cfg.hkv * num_block_slots)


def compile_keys_for_request(cfg: AotConfig, *, tq: int, num_block_slots: int) -> dict[str, tuple]:
    """The exact compile key of every kernel for one request. The C++ op computes
    these and asserts each is registered (else fatal -- never recompiles)."""
    r = replicas_for_request(cfg, tq=tq, num_block_slots=num_block_slots)
    return {
        "init": _key_init(cfg, r),
        "count": _key_count(cfg, r),
        "reduce": _key_reduce(r),
        "offsets": _key_offsets(cfg, parallel_offsets_for_request(num_block_slots)),
        "scatter": _key_scatter(cfg, r),
        "scheduler": _key_scheduler(cfg),
        "forward": _key_forward(cfg),
        "combine": _key_combine(cfg),
    }


# --------------------------------------------------------------------------- #
# Compile-template builders (one per kernel). Shapes are tiny placeholders --
# only dtype / rank / leading_dim affect the ABI and the (config-only) kernel.
# Each returns (kernel_obj, [positional template args incl. the fake stream]).
# --------------------------------------------------------------------------- #
def _dev() -> str:
    return "cuda"


def _t(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(*shape, dtype=dtype, device=_dev())


def _build_init(cfg: AotConfig, replicas: int):
    hkv, ratio = cfg.hkv, cfg.ratio
    nbs, tq, msb_cols = 16, 32, 8
    selected = _t(tq, hkv, cfg.topk, dtype=torch.int32)
    block_tables = _t(2, msb_cols, dtype=torch.int32) if cfg.has_block_tables else selected[0]
    count = _t(hkv, nbs, replicas, dtype=torch.int32)  # 3D: per-slot replica counters
    slot = _t(hkv, nbs * ratio, dtype=torch.int64)
    kernel = _InitSlotsAndCountsKernel(
        hkv=hkv,
        ratio=ratio,
        page_size=cfg.page_size,
        has_block_tables=cfg.has_block_tables,
        replicas=replicas,
    )
    args = [
        to_cute_tensor(selected, assumed_align=4, leading_dim=2),
        to_cute_tensor(block_tables, assumed_align=4, leading_dim=1),
        to_cute_tensor(count, assumed_align=4, leading_dim=2),
        to_cute_tensor(slot, assumed_align=8, leading_dim=1),
        Int32(1),
        Int32(1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    ]
    return kernel, args


def _build_count(cfg: AotConfig, replicas: int):
    hkv, ratio = cfg.hkv, cfg.ratio
    nbs, tq = 16, 32
    selected = _t(tq, hkv, cfg.topk, dtype=torch.int32)
    cuq = _t(2, dtype=torch.int64)
    sk = _t(1, dtype=torch.int32)
    slot = _t(hkv, nbs * ratio, dtype=torch.int64)
    count = _t(hkv, nbs, replicas, dtype=torch.int32)  # 3D
    edge_local = _t(tq * hkv * cfg.topk, dtype=torch.int32)
    kernel = _CountEdgesKernel(
        h_idx=hkv,
        topk=cfg.topk,
        block_size=cfg.block_size,
        causal=cfg.causal,
        has_block_tables=cfg.has_block_tables,
        ratio=ratio,
        replicas=replicas,
    )
    args = [
        to_cute_tensor(selected, assumed_align=4, leading_dim=2),
        to_cute_tensor(cuq, assumed_align=8, leading_dim=0),
        to_cute_tensor(sk, assumed_align=4, leading_dim=0),
        to_cute_tensor(slot, assumed_align=8, leading_dim=1),
        to_cute_tensor(count, assumed_align=4, leading_dim=2),
        to_cute_tensor(edge_local, assumed_align=4, leading_dim=0),
        Int32(1),
        Int32(1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    ]
    return kernel, args


def _build_reduce(cfg: AotConfig, replicas: int):
    hkv = cfg.hkv
    nbs = 16
    count = _t(hkv, nbs, replicas, dtype=torch.int32)  # 3D replica counters (in)
    count_total = _t(hkv, nbs, dtype=torch.int32)  # 2D per-slot total (out)
    kernel = _ReduceReplicasKernel(replicas=replicas)
    args = [
        to_cute_tensor(count, assumed_align=4, leading_dim=2),
        to_cute_tensor(count_total, assumed_align=4, leading_dim=1),
        Int32(1),  # num_units
        Int32(1),  # num_block_slots
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    ]
    return kernel, args


def _build_offsets(cfg: AotConfig, parallel: bool):
    # Operates on the 2D per-slot total (count_total) produced by reduce; replica-independent.
    hkv = cfg.hkv
    nbs = 16
    count_total = _t(hkv, nbs, dtype=torch.int32)
    offsets = _t(hkv, nbs + 1, dtype=torch.int32)
    # Fused selected-slot compaction outputs (see _CountToOffsets*Kernel).
    sel_slots = _t(hkv, nbs, dtype=torch.int32)
    sel_offsets = _t(hkv, nbs + 1, dtype=torch.int32)
    num_sel = _t(hkv, dtype=torch.int32)
    kernel = _make_count_to_offsets_kernel(hkv=hkv, parallel=parallel)
    args = [
        to_cute_tensor(count_total, assumed_align=4, leading_dim=1),
        to_cute_tensor(offsets, assumed_align=4, leading_dim=1),
        to_cute_tensor(sel_slots, assumed_align=4, leading_dim=1),
        to_cute_tensor(sel_offsets, assumed_align=4, leading_dim=1),
        to_cute_tensor(num_sel, assumed_align=4, leading_dim=0),
        Int32(1),  # num_block_slots
    ]
    if parallel:
        args.append(Int32(1))  # chunk_size
    args.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False))
    return kernel, args


def _build_scatter(cfg: AotConfig, replicas: int):
    hkv, ratio = cfg.hkv, cfg.ratio
    nbs, tq = 16, 32
    edge_local = _t(tq * hkv * cfg.topk, dtype=torch.int32)
    selected = _t(tq, hkv, cfg.topk, dtype=torch.int32)
    cuq = _t(2, dtype=torch.int64)
    sk = _t(1, dtype=torch.int32)
    slot = _t(hkv, nbs * ratio, dtype=torch.int64)
    offsets = _t(hkv, nbs + 1, dtype=torch.int32)
    count = _t(hkv, nbs, replicas, dtype=torch.int32)  # 3D replica exclusive-prefix base
    idx_ranks = _t(hkv, tq * cfg.topk, 2, dtype=torch.int32)
    inv = _t(hkv, tq, cfg.topk, dtype=torch.int32)
    kernel = _ScatterRanksKernel(
        h_idx=hkv,
        topk=cfg.topk,
        block_size=cfg.block_size,
        causal=cfg.causal,
        has_block_tables=cfg.has_block_tables,
        ratio=ratio,
        replicas=replicas,
    )
    args = [
        to_cute_tensor(edge_local, assumed_align=4, leading_dim=0),
        to_cute_tensor(selected, assumed_align=4, leading_dim=2),
        to_cute_tensor(cuq, assumed_align=8, leading_dim=0),
        to_cute_tensor(sk, assumed_align=4, leading_dim=0),
        to_cute_tensor(slot, assumed_align=8, leading_dim=1),
        to_cute_tensor(offsets, assumed_align=4, leading_dim=1),
        to_cute_tensor(count, assumed_align=4, leading_dim=2),
        to_cute_tensor(idx_ranks, assumed_align=4, leading_dim=2),
        to_cute_tensor(inv, assumed_align=4, leading_dim=2),
        Int32(1),
        Int32(1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    ]
    return kernel, args


def _build_scheduler(cfg: AotConfig):
    hkv, num_splits, nbs = cfg.hkv, cfg.num_splits, 16
    offs = _t(hkv, nbs + 1, dtype=torch.int32)
    ws = _t(num_splits, 3, dtype=torch.int32)
    we = _t(num_splits, 3, dtype=torch.int32)
    kernel = _LoadBalanceScheduler(hkv, num_splits)
    args = [
        to_cute_tensor(offs, assumed_align=4, leading_dim=1),
        to_cute_tensor(ws, assumed_align=4, leading_dim=1),
        to_cute_tensor(we, assumed_align=4, leading_dim=1),
        Int32(1),
        Int64(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    ]
    return kernel, args


def _build_forward(cfg: AotConfig):
    hkv, qhead, d = cfg.hkv, cfg.qhead, cfg.head_dim
    hq = hkv * qhead
    tq, nbs, num_pages, ps = 32, 16, 4, cfg.page_size
    topk = cfg.topk
    qls, obuf = _qls_obuf(cfg)
    qdt, odt, pdt = cfg.q_torch(), cfg.out_torch(), cfg.partial_torch()

    q = _t(tq, hq, d, dtype=qdt)
    k_cache = _t(num_pages, hkv, ps, d, dtype=qdt)
    v_cache = _t(num_pages, hkv, ps, d, dtype=qdt)
    k_perm = k_cache.permute(0, 2, 1, 3)
    v_perm = v_cache.permute(0, 2, 1, 3)
    seg = tq * topk * qhead
    o_flat = _t(hkv * tq * topk * qhead, d, dtype=pdt)
    m_partial = _t(hkv, seg, dtype=torch.float32)
    l_partial = _t(hkv, seg, dtype=torch.float32)
    slot = _t(hkv, nbs * cfg.ratio, dtype=torch.int64)
    # The compact forward consumes the COMPACT CSR (sel_offsets) + sel_slots/num_sel, not the
    # dense offsets (see indexed_block_partials); the scheduler emits compact-j block indices.
    sel_offsets = _t(hkv, nbs + 1, dtype=torch.int32)
    sel_slots = _t(hkv, nbs, dtype=torch.int32)
    num_sel = _t(hkv, dtype=torch.int32)
    idx_ranks = _t(hkv, tq * topk, 2, dtype=torch.int32)
    ws = _t(cfg.num_splits, 3, dtype=torch.int32)
    we = _t(cfg.num_splits, 3, dtype=torch.int32)
    cuq = _t(2, dtype=torch.int64)
    sk = _t(1, dtype=torch.int32)

    kernel = SparseKVOuterForward(
        qhead,
        hkv,
        cfg.page_size,
        causal=cfg.causal,
        q_load_stage=qls,
        o_buffers=obuf,
    )
    o2d_t = to_cute_tensor(o_flat, leading_dim=1)
    args = [
        to_cute_tensor(q, leading_dim=2),
        to_cute_tensor(k_perm, leading_dim=3),
        to_cute_tensor(v_perm, leading_dim=3),
        o2d_t,  # mO (only element_type used)
        to_cute_tensor(m_partial, assumed_align=4, leading_dim=1),
        to_cute_tensor(l_partial, assumed_align=4, leading_dim=1),
        to_cute_tensor(slot, assumed_align=8, leading_dim=1),
        to_cute_tensor(sel_offsets, assumed_align=4, leading_dim=1),  # COMPACT CSR (mKvToQOffsets)
        to_cute_tensor(idx_ranks, assumed_align=4, leading_dim=2),
        to_cute_tensor(ws, assumed_align=4, leading_dim=1),
        to_cute_tensor(we, assumed_align=4, leading_dim=1),
        to_cute_tensor(sel_slots, assumed_align=4, leading_dim=1),  # mSelSlots
        to_cute_tensor(num_sel, assumed_align=4, leading_dim=0),  # mNumSel
        Int32(1),  # grid_size
        to_cute_tensor(cuq, assumed_align=8, leading_dim=0),
        to_cute_tensor(sk, assumed_align=4, leading_dim=0),
        Int32(1),  # n_batches
        Float32(1.0),  # softmax_scale
        o2d_t,  # mO2d
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    ]
    return kernel, args


def _build_combine(cfg: AotConfig):
    # lazy: combine is only needed while exporting its AOT kernel.
    from .sparse_fwd_kvouter_combine import (
        FlashAttentionForwardCombine,
        _TORCH2CUTE,
    )

    hkv, qhead, d, topk = cfg.hkv, cfg.qhead, cfg.head_dim, cfg.topk
    hq = hkv * qhead
    tq = 32
    out_dt, pdt = cfg.out_torch(), cfg.partial_torch()
    r_total = hkv * tq * topk * qhead
    seg = tq * topk * qhead

    o_partial = _t(r_total, d, dtype=pdt)
    lse_partial = _t(hkv * seg, dtype=torch.float32)  # flat 1D
    l_partial = _t(hkv * seg, dtype=torch.float32)
    inv = _t(hkv, tq, topk, dtype=torch.int32)
    out = _t(1, tq, hq, d, dtype=out_dt)  # batched (Tq,Hq,D) -> unsqueeze(0)
    lse = _t(1, hq, tq, dtype=torch.float32) if cfg.return_lse else None

    log_max_splits = max(math.ceil(math.log2(max(topk, 2))), 5)
    num_threads = 128
    k_block_size = 64 if d <= 64 else 128
    k_block_gmem = 128 if k_block_size % 128 == 0 else (64 if k_block_size % 64 == 0 else 32)
    async_copy_elems = 128 // _TORCH2CUTE[pdt].width
    tile_m = num_threads * async_copy_elems // k_block_gmem
    kernel = FlashAttentionForwardCombine(
        dtype=_TORCH2CUTE[out_dt],
        dtype_partial=_TORCH2CUTE[pdt],
        head_dim=d,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
        num_threads=num_threads,
    )
    op_t = to_cute_tensor(o_partial, assumed_align=16, leading_dim=1)
    lp_t = to_cute_tensor(lse_partial, assumed_align=4, leading_dim=0)
    l_t = to_cute_tensor(l_partial, assumed_align=4, leading_dim=0)
    inv_t = to_cute_tensor(inv, assumed_align=4, leading_dim=2)
    o_t = to_cute_tensor(out, assumed_align=16, leading_dim=3)
    lse_t = to_cute_tensor(lse, assumed_align=4, leading_dim=2) if cfg.return_lse else None
    # __call__ order: mO_partial, mLSE_partial, mO, mL_partial, mInv, mLSE,
    # cu_seqlens, seqused, num_splits_dynamic_ptr, varlen_batch_idx,
    # semaphore_to_reset, stream
    args = [
        op_t,
        lp_t,
        o_t,
        l_t,
        inv_t,
        lse_t,
        None,
        None,
        None,
        None,
        None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    ]
    return kernel, args


# init/count/reduce/scatter take an extra `replicas` builder arg; offsets takes `parallel`.
_BUILDERS: dict[str, Callable[..., tuple]] = {
    "init": _build_init,
    "count": _build_count,
    "reduce": _build_reduce,
    "offsets": _build_offsets,
    "scatter": _build_scatter,
    "scheduler": _build_scheduler,
    "forward": _build_forward,
    "combine": _build_combine,
}


# --------------------------------------------------------------------------- #
# Export + cache
# --------------------------------------------------------------------------- #
# Per-process export dir. Default: a FRESH temp dir created once per process, so every
# restart recompiles from scratch -- this guarantees a stale .o is never reused across a
# kernel-source / toolkit / config change (the cache key can't capture source edits). It
# is removed at process exit. The compile cost is a one-time warmup per config, identical
# to the JIT path's first-call cost; we just don't persist it across restarts.
#
# Set MINIMAX_KERNELS_CUTE_AOT_CACHE to a fixed path to opt into a PERSISTENT cache (faster restarts,
# but you then own invalidation on upgrades).
_CACHE_DIR: Optional[Path] = None


def _cache_dir() -> Path:
    global _CACHE_DIR
    root = os.environ.get("MINIMAX_KERNELS_CUTE_AOT_CACHE")
    if root:
        return Path(root)
    if _CACHE_DIR is None:
        _CACHE_DIR = Path(tempfile.mkdtemp(prefix="fmha_sm100_cute_aot_"))
        atexit.register(shutil.rmtree, str(_CACHE_DIR), True)  # ignore_errors
    return _CACHE_DIR


def _arch_tag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def _key_hash(kernel: str, key: tuple) -> str:
    import cutlass

    payload = json.dumps(
        [kernel, list(key), getattr(cutlass, "__version__", "?"), _arch_tag()],
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _compile_and_dump(kernel_obj: Any, template_args: list, prefix: str) -> tuple[bytes, list[ArgDesc]]:
    global _AOT_ALLOWED
    _AOT_ALLOWED = True
    try:
        compiled = cute.compile(kernel_obj, *template_args)
    finally:
        _AOT_ALLOWED = False
    # Derive the C-ABI descriptors from the template args directly (positional). The
    # C++ packer consumes args by ORDER + masks, not by name, so we don't need the
    # compiled object's arg-name spec -- avoiding a version-fragile internal
    # (older/newer CuTe DSL builds don't expose `.args_spec`).
    descs: list[ArgDesc] = []
    for i, val in enumerate(template_args):
        d = _classify(f"arg{i}", val)
        if d is not None:
            descs.append(d)
    obj_bytes = compiled.dump_to_object(prefix)
    return obj_bytes, descs


def _export_one(cfg: AotConfig, kernel: str, key: tuple, builder_args: tuple = ()) -> KernelArtifact:
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    khash = _key_hash(kernel, key)
    prefix = f"{kernel}_{khash}"
    obj_path = cache / f"{prefix}.o"
    meta_path = cache / f"{prefix}.json"

    if obj_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        args = [ArgDesc(**a) for a in meta["args"]]
        return KernelArtifact(kernel, list(key), prefix, str(obj_path), args)

    builder = _BUILDERS[kernel]
    kernel_obj, template_args = builder(cfg, *builder_args)
    obj_bytes, descs = _compile_and_dump(kernel_obj, template_args, prefix)
    tmp = obj_path.with_suffix(".o.tmp")
    tmp.write_bytes(obj_bytes)
    os.replace(tmp, obj_path)
    art = KernelArtifact(kernel, list(key), prefix, str(obj_path), descs)
    meta_path.write_text(json.dumps({"key": list(key), "args": [asdict(a) for a in descs]}, default=str))
    return art


def export_all_reachable(cfg: AotConfig) -> dict[str, KernelArtifact]:
    """Compile + dump every kernel in the full reachable key set for ``cfg``.

    Two request-variable axes are fully enumerated so neither ever triggers a
    recompile at runtime:
      * offsets parallel/serial split (by ``num_block_slots``) -> slots
        ``offsets:parallel`` / ``offsets:serial``.
      * adaptive index ``replicas`` (by ``tq``/nbins; every reachable value in
        ``_replica_values()``) -> the per-replica index kernels are keyed
        ``init:r<R>`` / ``count:r<R>`` / ``reduce:r<R>`` / ``scatter:r<R>``.
    The C++ op selects the right slot from ``num_block_slots`` and ``replicas``.
    Returns a map from logical slot name to :class:`KernelArtifact`.
    """
    out: dict[str, KernelArtifact] = {}
    # Replica-independent kernels (once).
    out["scheduler"] = _export_one(cfg, "scheduler", _key_scheduler(cfg))
    out["forward"] = _export_one(cfg, "forward", _key_forward(cfg))
    out["combine"] = _export_one(cfg, "combine", _key_combine(cfg))
    out["offsets:serial"] = _export_one(cfg, "offsets", _key_offsets(cfg, False), (False,))
    out["offsets:parallel"] = _export_one(cfg, "offsets", _key_offsets(cfg, True), (True,))
    # Per-replica index kernels (init/count/reduce/scatter) over all reachable R.
    for r in _replica_values():
        out[f"init:r{r}"] = _export_one(cfg, "init", _key_init(cfg, r), (r,))
        out[f"count:r{r}"] = _export_one(cfg, "count", _key_count(cfg, r), (r,))
        out[f"reduce:r{r}"] = _export_one(cfg, "reduce", _key_reduce(r), (r,))
        out[f"scatter:r{r}"] = _export_one(cfg, "scatter", _key_scatter(cfg, r), (r,))
    return out
