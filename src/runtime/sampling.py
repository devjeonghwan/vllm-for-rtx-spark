"""Fuse the ordinary top-k verifier's FP32 logits buffer when it is unobserved."""

import numpy as np


def eligible(owner, logits, idx):
    sampler = owner.sampler
    # Chunk processing passes a batch-global logprobs request to every chunk.
    # A chunk without local requests can still have a processed-logits consumer.
    if sampler.logprobs_mode in ("processed_logprobs", "processed_logits"):
        return False, "processed_logprobs_mode"
    if not owner.use_block_verification or owner.synthetic_conditional_rates is not None:
        return False, "verification_mode"
    if owner.watermark_key is not None:
        return False, "watermark"
    if getattr(sampler, "trace_replay_state", None) is not None:
        return False, "trace_replay"
    if sampler.return_sampling_mask:
        return False, "sampling_mask"
    if logits.ndim != 2 or logits.stride(-1) != 1 or logits.device.type != "cuda" or logits.shape[0] == 0:
        return False, "layout"
    if str(logits.dtype) not in ("torch.float32", "torch.float16", "torch.bfloat16"):
        return False, "dtype"
    states = sampler.sampling_states
    if np.any(states.num_logprobs[idx] != -1):
        return False, "requested_logprobs"
    k, p, t = states.top_k.np[idx], states.top_p.np[idx], states.temperature.np[idx]
    if np.any(k <= 0) or np.any(k >= logits.shape[1]):
        return False, "top_k_inactive"
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        return False, "greedy_or_invalid_temperature"
    if np.any(~np.isfinite(p)) or np.any(p <= 0) or np.any(p > 1):
        return False, "invalid_top_p"
    if np.any(states.min_p.np[idx] != 0):
        return False, "min_p"
    processors = sampler.logits_processors
    expected = (
        ("LogitBiasState", "use_logit_bias"),
        ("PenaltiesState", "use_penalty"),
        ("BadWordsState", "num_bad_words"),
    )
    if len(processors) != len(expected):
        return False, "custom_processors"
    for processor, (name, attribute) in zip(processors, expected):
        if type(processor).__name__ != name or not type(processor).__module__.startswith(
            "vllm.v1.worker.gpu.sample."
        ):
            return False, "custom_processors"
        values = getattr(processor, attribute)
        if attribute == "num_bad_words":
            values = values.np
        if np.any(values[idx]):
            return False, name
    budget = sampler.thinking_budget_state
    if budget.enabled and np.any(budget.use_thinking_budget[idx]):
        return False, "thinking_budget"
    return True, "eligible"


def try_verify(
    owner,
    logits,
    draft_logits,
    draft_sampled,
    pos,
    cu_num_logits,
    idx_mapping,
    idx_mapping_np,
    expanded_idx_mapping,
    expanded_local_pos,
    seq_lens_upper_bound_np,
    verify_draft_sampled,
):
    """Return the fused result, or None so the original verifier runs unchanged.

    Kernels import on the first supported call, after normal worker startup.
    The mask is a local tensor produced and consumed on the current CUDA stream.
    Qrita's original FP32 search scratch remains owned by its existing cache.
    """
    if not eligible(owner, logits, idx_mapping_np)[0]:
        return None
    from ..kernels.sampling import make_filter, masked_rejection

    states = owner.sampler.sampling_states
    k, p = states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
    bits = make_filter(logits, expanded_idx_mapping, states.temperature.gpu, k, p)
    sampled, num_sampled = masked_rejection(
        logits,
        draft_logits,
        verify_draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        states.temperature.gpu,
        states.seeds.gpu,
        owner.num_speculative_steps,
        bits,
        use_fp64=owner.sampler.use_fp64_gumbel,
        use_block_verification=True,
    )
    # No supported caller consumes processed logits, logprobs or a sampling mask.
    return logits, sampled, num_sampled
