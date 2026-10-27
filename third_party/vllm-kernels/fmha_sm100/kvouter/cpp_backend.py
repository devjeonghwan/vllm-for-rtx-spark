# Copyright (c) 2026 Fireworks AI
# SPDX-License-Identifier: Apache-2.0

"""Python entry point for the C++ KV-outer sparse-attention backend.

On the first call for a configuration this AOT-exports every reachable CuTe-DSL
kernel and initializes the package extension with the object paths and config.
Subsequent calls dispatch directly to the C++ op.

``FMHA_SM100_KVOUTER_CPP=0`` forces the Python backend and ``=1`` requires
this backend. ``MINIMAX_KERNELS_KVOUTER_CPP`` is accepted as a legacy alias.
"""

from __future__ import annotations

import ctypes
import importlib
import math
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

from .aot_export import (
    AotConfig,
    export_all_reachable,
    replicas_for_request,
)
from .build_kvouter_index import (
    _COUNT_TO_OFFSETS_PARALLEL_THRESHOLD,
)

__all__ = ["kvouter_attention_cpp", "cpp_backend_available"]

_DTYPE_CODE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
_TORCH_TO_AOT = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float8_e4m3fn: "fp8e4m3",
    torch.float8_e5m2: "fp8e5m2",
}
_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)

# AotConfig -> opaque C++ handle id (one init per deployment config). Guarded by
# _HANDLE_LOCK so concurrent first-callers don't both run the (expensive) AOT export +
# C++ init and leak duplicate handles. Keyed by (AotConfig, cuda device index): the C++
# init loads the kernels into a per-device CUDA context (and on a mixed-arch host the
# kernels would differ), so each device gets its own handle.
_HANDLE_CACHE: dict[tuple[AotConfig, int], int] = {}
_HANDLE_LOCK = threading.Lock()
_EXTENSION_LOAD_ERROR: Optional[BaseException] = None


def _jit_build_extension() -> None:
    """Build the C++ op with torch.utils.cpp_extension when ``fmha_sm100._C`` is
    absent. vLLM installs fmha_sm100 as plain files (no setup.py build), so the
    op is compiled once per host and cached under ~/.cache/torch_extensions."""
    import nvidia_cutlass_dsl
    from torch.utils.cpp_extension import CUDA_HOME, load

    csrc = Path(__file__).resolve().parent.parent / "csrc" / "kvouter"
    cute_base = next(
        Path(p)
        for p in nvidia_cutlass_dsl.__path__
        if (Path(p) / "include" / "CuteDSLRuntime.h").is_file()
        and (Path(p) / "lib" / "libcute_dsl_runtime.so").is_file()
    )
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME/nvcc not found; a CUDA 13 toolkit is required")
    cuda_home = Path(CUDA_HOME)
    stub_dirs = [
        d
        for d in (
            cuda_home / "lib64" / "stubs",
            cuda_home / "lib" / "stubs",
            *sorted((cuda_home / "targets").glob("*/lib/stubs")),
        )
        if (d / "libcuda.so").is_file()
    ]
    lib = cute_base / "lib"
    load(
        name="fmha_sm100_kvouter_C",
        sources=[str(csrc / "bindings.cpp"), str(csrc / "cute_sparse_kvouter.cpp")],
        extra_include_paths=[str(csrc), str(cute_base / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[
            f"-L{lib}",
            f"-Wl,-rpath,{lib}",
            "-lcute_dsl_runtime",
            *(f"-L{d}" for d in stub_dirs),
            "-lcuda",
        ],
        with_cuda=True,
        verbose=False,
    )


def _ensure_extension_loaded() -> bool:
    """Load the CuTe runtime globally, then register the package operators."""
    global _EXTENSION_LOAD_ERROR
    if hasattr(torch.ops.fmha_sm100, "sparse_kvouter_attn"):
        return True
    try:
        # lazy: CuTe-DSL is only needed when probing or using the C++ backend.
        import cutlass.cute as cute

        for path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        try:
            importlib.import_module("fmha_sm100._C")
        except ImportError:
            _jit_build_extension()
    except Exception as exc:
        _EXTENSION_LOAD_ERROR = exc
        return False
    return hasattr(torch.ops.fmha_sm100, "sparse_kvouter_attn")


def _ops() -> object:
    if not _ensure_extension_loaded():
        raise RuntimeError(
            "fmha_sm100 C++ op 'sparse_kvouter_attn' is unavailable; "
            "reinstall fmha_sm100 with its CUDA extension enabled"
        ) from _EXTENSION_LOAD_ERROR
    return torch.ops.fmha_sm100


def cpp_backend_available() -> bool:
    """Return whether the package's focused C++ extension is loadable."""
    return _ensure_extension_loaded()


def _derive_config(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    selected: torch.Tensor,
    *,
    block_size: int,
    page_size: int,
    causal: bool,
    out_dtype: torch.dtype,
    partial_dtype: Optional[torch.dtype],
    store_in_corr: bool,
    return_lse: bool,
    num_splits: int,
) -> AotConfig:
    nheads_kv = k_cache.shape[1]
    qhead = q.shape[1] // nheads_kv
    q_is_fp8 = q.dtype in _FP8
    if partial_dtype is None:
        partial_dtype = torch.bfloat16 if q_is_fp8 else q.dtype
    # The C++ op allocates O_partial / output / LSE in these dtypes, so partial and out
    # must be among the runtime-supported set (q/k/v may still be fp8 — they're passed
    # through as raw bytes and never allocated here).
    for name, dt in (("partial_dtype", partial_dtype), ("out_dtype", out_dtype)):
        if dt not in _DTYPE_CODE:
            supported = ", ".join(str(d) for d in _DTYPE_CODE)
            raise ValueError(
                f"cute kvouter C++ backend: {name}={dt} is unsupported; "
                f"expected one of [{supported}] (fp8 partial/output is not supported)"
            )
    return AotConfig(
        nheads_kv=nheads_kv,
        qhead=qhead,
        topk=selected.shape[-1],
        block_size=block_size,
        page_size=page_size,
        causal=bool(causal),
        head_dim=q.shape[-1],
        num_splits=num_splits,
        q_dtype=_TORCH_TO_AOT[q.dtype],
        out_dtype=_TORCH_TO_AOT[out_dtype],
        partial_dtype=_TORCH_TO_AOT[partial_dtype],
        store_in_corr=bool(store_in_corr),
        has_block_tables=True,
        return_lse=bool(return_lse),
    )


def _get_or_init_handle(cfg: AotConfig, device: torch.device) -> int:
    dev = torch.device(device)
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (cfg, dev_idx)
    cached = _HANDLE_CACHE.get(key)
    if cached is not None:
        return cached
    # Double-checked lock: serialize the first init for a given (cfg, device) so
    # concurrent callers don't duplicate the AOT export + C++ init (and leak handles).
    with _HANDLE_LOCK:
        cached = _HANDLE_CACHE.get(key)
        if cached is not None:
            return cached
        # lazy: CuTe-DSL export is only needed for first-time AOT initialization.
        import cutlass.cute as cute

        # export_all_reachable() runs cute.compile, which targets the *current* CUDA
        # device's arch, and sparse_kvouter_init loads the kernels into that device's
        # CUDA context. The op later launches on q.device() (pinned by a CUDAGuard in
        # C++), so pin the current device to q.device() here -- otherwise on a
        # mixed-arch host the cached kernels could be built for / loaded on the wrong GPU.
        with torch.cuda.device(dev_idx):
            arts = export_all_reachable(cfg)
            slots = list(arts.keys())  # includes per-replica index slots (init:r16, ...)
            paths = [arts[s].object_path for s in slots]
            prefixes = [arts[s].function_prefix for s in slots]
            runtime_libs = list(cute.runtime.find_runtime_libraries(enable_tvm_ffi=False))
            handle = _ops().sparse_kvouter_init(
                slots,
                paths,
                prefixes,
                runtime_libs,
                cfg.topk,
                cfg.block_size,
                cfg.page_size,
                cfg.num_splits,
                cfg.return_lse,
                _DTYPE_CODE[cfg.partial_torch()],
                _DTYPE_CODE[cfg.out_torch()],
                _COUNT_TO_OFFSETS_PARALLEL_THRESHOLD,
            )
        _HANDLE_CACHE[key] = handle
        return handle


def kvouter_attention_cpp(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    used_kv_lens: Optional[torch.Tensor] = None,
    block_size: int = 128,
    page_size: int = 64,
    out_dtype: torch.dtype = torch.bfloat16,
    return_lse: bool = False,
    partial_dtype: Optional[torch.dtype] = None,
    store_in_corr: bool = True,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """C++-backed equivalent of :func:`...interface.kvouter_attention`.

    Same inputs/outputs; the entire index-build + forward + combine pipeline runs in
    the package C++ op against AOT-compiled kernels.
    """
    assert cu_seqlens_q is not None, "cu_seqlens_q is required; use [0, Tq] for a single sequence"
    device = q.device
    d = q.shape[-1]
    # The AOT-exported forward kernel is fixed to 128-token blocks and head_dim=128
    # (see sparse_fwd_kvouter: m/n_block_size=128, head_dim=128). Other values would
    # compile mismatched index kernels and silently yield wrong attention, so reject
    # them up front rather than producing garbage. Mirrors interface.BLOCK_SIZE/HEAD_DIM
    # (not imported here to avoid an interface <-> cpp_backend import cycle).
    if block_size != 128:
        raise ValueError(
            f"cute kvouter C++ backend: block_size={block_size} is unsupported; "
            "the AOT-exported forward kernel only supports block_size=128"
        )
    if d != 128:
        raise ValueError(
            f"cute kvouter C++ backend: head_dim={d} is unsupported; "
            "the AOT-exported forward kernel only supports head_dim=128"
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    ratio = block_size // page_size
    msb = max(1, block_tables.shape[1] // ratio)
    n_batches = cu_seqlens_q.shape[0] - 1
    cu_seqlens_q_i64 = cu_seqlens_q.to(device=device, dtype=torch.int64).contiguous()
    if used_kv_lens is None:
        # Mirror build_kvouter_index's default (uniform msb * block_size).
        used_kv_lens = torch.full((n_batches,), msb * block_size, dtype=torch.int32, device=device)
    else:
        used_kv_lens = used_kv_lens.to(device=device, dtype=torch.int32).contiguous()

    num_splits = torch.cuda.get_device_properties(device).multi_processor_count
    cfg = _derive_config(
        q,
        k_cache,
        selected,
        block_size=block_size,
        page_size=page_size,
        causal=causal,
        out_dtype=out_dtype,
        partial_dtype=partial_dtype,
        store_in_corr=store_in_corr,
        return_lse=return_lse,
        num_splits=num_splits,
    )
    handle = _get_or_init_handle(cfg, device)
    # Adaptive index replica count (request-variable; selects the matching pre-exported
    # init/count/reduce/scatter kernels in the C++ op). num_block_slots = B*msb.
    num_block_slots = msb if n_batches == 1 else n_batches * msb
    replicas = replicas_for_request(cfg, tq=q.shape[0], num_block_slots=num_block_slots)
    o, lse = _ops().sparse_kvouter_attn(
        handle,
        q,
        k_cache,
        v_cache,
        selected.contiguous(),
        block_tables,
        cu_seqlens_q_i64,
        used_kv_lens,
        float(softmax_scale),
        int(replicas),
        out,
    )
    if not return_lse:
        return o, None
    return o, lse
