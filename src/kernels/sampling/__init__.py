"""Packed top-k/top-p selection and temperature-scaled rejection loads.

The original search scratch, RNG and verification algorithms are retained.
Imports allocate no GPU memory; generated Triton modules load on first use.
"""

_TOPK = None
_REJECTION = None


def make_filter(logits, mapping, temperatures, k, p):
    import torch
    from vllm.v1.sample.ops import topk_topp_triton as original
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed
    from vllm.utils.math_utils import next_power_of_2
    from vllm.utils.platform_utils import num_compute_units

    global _TOPK
    if _TOPK is None:
        from . import topk

        _TOPK = topk
    rows, vocab = logits.shape
    programs = min(num_compute_units(logits.device.index), rows)
    buffer_key = (logits.device, torch.float32, vocab)
    buffer = original._TRITON_BUFFER_CACHE.get(buffer_key)
    if buffer is None or buffer.shape[0] < programs:
        size = min(next_power_of_2(programs), num_compute_units(logits.device.index))
        buffer = torch.empty((size, vocab), dtype=torch.float32, device=logits.device)
        original._TRITON_BUFFER_CACHE[buffer_key] = buffer
    buffer = buffer[:programs]
    tables = original._TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        with gpu_sync_allowed():
            tables = (
                torch.tensor(original._NORMAL_CDF_TO_SIGMA_TABLE, dtype=torch.float32, device=logits.device),
                torch.tensor(original._PERCENTILE_TO_STD_TABLE, dtype=torch.float32, device=logits.device),
            )
        original._TRITON_TABLE_CACHE[logits.device] = tables
    normal, percentile = tables
    words = (vocab + 31) // 32
    bits = torch.empty((rows, words), dtype=torch.int32, device=logits.device)
    _TOPK._rtx_topk_topp_mask_kernel[(programs,)](
        logits,
        logits.stride(0),
        buffer,
        percentile,
        normal,
        k,
        p if p is not None else logits,
        rows,
        VOCAB_SIZE=vocab,
        MASK_VALUE=-float("inf"),
        BLOCK_SIZE=8192,
        BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=True,
        TOPP_ENABLED=p is not None,
        SPLIT_COVERS_PONLY=False,
        num_warps=8,
        _rtx_filter=bits,
        _rtx_filter_stride=words,
        _rtx_mapping=mapping,
        _rtx_temperature=temperatures,
    )
    return bits


def masked_rejection(
    logits,
    draft_logits,
    draft_sampled,
    cu_num_logits,
    pos,
    idx_mapping,
    expanded_idx_mapping,
    expanded_local_pos,
    temperature,
    seeds,
    num_speculative_steps,
    bits,
    *,
    use_fp64=False,
    use_block_verification=True,
):
    global _REJECTION
    if _REJECTION is None:
        from . import rejection

        _REJECTION = rejection
    return _REJECTION.rejection_sample(
        logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seeds,
        num_speculative_steps,
        None,
        use_fp64=use_fp64,
        use_block_verification=use_block_verification,
        _rtx_filter=bits,
        _rtx_filter_stride=bits.stride(0),
    )
