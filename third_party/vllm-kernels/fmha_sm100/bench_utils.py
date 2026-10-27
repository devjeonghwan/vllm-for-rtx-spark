# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""GPU benchmarking utilities for FMHA SM100 kernels.

Provides bench_gpu_time() using CUDA events with L2 cache flushing.
No dependency on flashinfer.
"""

import torch
import numpy as np
from typing import Callable, List, Optional, Tuple


def get_l2_cache_size(device: str = "cuda") -> int:
    """Get L2 cache size in bytes for the given CUDA device."""
    props = torch.cuda.get_device_properties(device)
    if hasattr(props, 'l2_cache_size'):
        return props.l2_cache_size
    return 256 * 1024 * 1024  # 256MB fallback


def bench_gpu_time(
    fn: Callable,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    cold_l2_cache: bool = True,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
) -> List[float]:
    """Benchmark kernel execution time using CUDA events.

    Args:
        fn: Kernel function to benchmark (called as fn() or fn(*input_args, **input_kwargs)).
        dry_run_time_ms: Target warmup duration in ms.
        repeat_time_ms: Target measurement duration in ms.
        dry_run_iters: Explicit warmup iteration count (overrides dry_run_time_ms).
        repeat_iters: Explicit measurement iteration count (overrides repeat_time_ms).
        cold_l2_cache: If True, flush L2 cache before each iteration.
        input_args: Positional arguments to pass to fn.
        input_kwargs: Keyword arguments to pass to fn.

    Returns:
        List of per-iteration execution times in milliseconds.
    """
    if input_kwargs is None:
        input_kwargs = {}

    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    # L2 cache flush buffer
    buffer = None
    if cold_l2_cache:
        device = "cuda"
        for arg in input_args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                device = str(arg.device)
                break
        l2_size = get_l2_cache_size(device)
        flush_size = l2_size * 2
        buffer = torch.empty(flush_size, device=device, dtype=torch.int8)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Estimate kernel time with 5 iterations
    torch.cuda.synchronize()
    call_fn()
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(5):
        if buffer is not None:
            buffer.zero_()
        call_fn()
    end_event.record()
    torch.cuda.synchronize()
    est_time = start_event.elapsed_time(end_event) / 5

    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / est_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / est_time))

    # Warmup
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if buffer is not None:
            buffer.zero_()
        call_fn()
    torch.cuda.synchronize()

    # Measure
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for i in range(repeat_iters):
        if buffer is not None:
            buffer.zero_()
        start_events[i].record()
        call_fn()
        end_events[i].record()

    torch.cuda.synchronize()
    return [start_events[i].elapsed_time(end_events[i]) for i in range(repeat_iters)]


def attention_tflops(
    qo_lens, kv_lens, head_dim_qk, head_dim_vo,
    num_qo_heads, causal, time_ms,
) -> float:
    """Compute attention TFLOP/s from actual per-batch sequence lengths."""
    if isinstance(qo_lens, torch.Tensor):
        qo_lens = qo_lens.cpu().tolist()
    if isinstance(kv_lens, torch.Tensor):
        kv_lens = kv_lens.cpu().tolist()

    total_flops = 0
    for ql, kl in zip(qo_lens, kv_lens):
        if causal:
            qk_flops = ql * kl - ql * (ql - 1) / 2
        else:
            qk_flops = ql * kl
        total_flops += num_qo_heads * qk_flops * (head_dim_qk + head_dim_vo) * 2

    return total_flops / (time_ms * 1e-3) / 1e12
