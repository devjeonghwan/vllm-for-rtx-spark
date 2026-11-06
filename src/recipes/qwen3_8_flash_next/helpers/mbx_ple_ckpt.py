# mbx_ple_ckpt — prefill checkpoints for the Qwen4Exp PLE short conv (patch 58), so vLLM's align-mode prefix cache can
# resume mid-prompt on Qwen3.8 (PR #60050 gives the GDN layers theirs, but takes checkpoints only when EVERY Mamba layer
# can export one, and the PLE short conv could not). Off unless MBX_PLE_PREFILL_CKPT=1.
# The PLE state is the last STATE_LEN = (kernel-1)*dilation conv inputs, kept in window columns [0, STATE_LEN) of the
# (canonicalized, capacity-sliced) conv-state slot — the layout _ple_conv_writeback_kernel writes after a prefill. The
# checkpoint is the same window taken at the checkpoint position instead of the chunk end, written into the checkpoint's
# own slot. Inputs are addressed exactly like the prefill conv: logical index q_start + i, mapped through the non-spec
# token map when the batch mixes spec and non-spec rows.
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

ON = os.environ.get("MBX_PLE_PREFILL_CKPT", "0") == "1"


@triton.jit
def _ple_checkpoint_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    offset_ptr,
    qsl_ptr,
    token_idx_ptr,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_MAP: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    r = tl.program_id(0)
    c_offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    sid = tl.load(state_idx_ptr + r).to(tl.int64)
    off = tl.load(offset_ptr + r).to(tl.int64)
    # off == 0: this row has no checkpoint. off < STATE_LEN cannot pass vLLM's validity rule (offset >= one hash
    # block), but a window reaching before the chunk would read other rows' tokens: never write a partial window.
    valid = (sid != NULL_STATE_ID) & (off >= STATE_LEN)
    mask = (c_offs < C) & valid
    q_start = tl.load(qsl_ptr + r).to(tl.int64)
    for h in tl.static_range(0, STATE_LEN):
        logical = q_start + off - STATE_LEN + h
        if HAS_MAP:
            phys = tl.load(token_idx_ptr + logical, mask=valid, other=0).to(tl.int64)
        else:
            phys = logical
        value = tl.load(x_ptr + phys * C + c_offs, mask=mask, other=0.0)
        tl.store(state_ptr + sid * state_bs + h * state_ws + c_offs * state_cs, value, mask=mask)


def export(inputs, conv_state, token_indices, query_start_loc, checkpoint, state_len: int) -> None:
    """Write each prefill row's checkpoint window. `checkpoint` rows align 1:1 with the prefill rows (and
    `query_start_loc`), `conv_state` is [slot, channel, window] as the prefill conv got it."""
    n = checkpoint.checkpoint_offsets.numel()
    if n == 0:
        return
    C = inputs.shape[1]
    block_c = 256
    state_bs, state_cs, state_ws = conv_state.stride()
    _ple_checkpoint_kernel[(n, triton.cdiv(C, block_c))](
        inputs,
        conv_state,
        checkpoint.state_indices,
        checkpoint.checkpoint_offsets,
        query_start_loc,
        token_indices if token_indices is not None else checkpoint.state_indices,
        state_bs,
        state_ws,
        state_cs,
        C=C,
        BLOCK_C=block_c,
        STATE_LEN=state_len,
        HAS_MAP=token_indices is not None,
        NULL_STATE_ID=NULL_BLOCK_ID,
    )


def prefill_rows(m, use_spec_decode: bool, num_decode_draft_tokens_cpu) -> list[int]:
    """The batch rows the PLE prefill conv runs, in its order: non-spec rows with query_len > 1 (the builder's own
    spec mask and decode/prefill split, ascending request order = the order of state_indices_p / query_start_loc_p)."""
    qlens = m.query_start_loc_cpu.diff().tolist()
    spec = [False] * m.num_reqs
    if use_spec_decode and num_decode_draft_tokens_cpu is not None:
        cand = (num_decode_draft_tokens_cpu[: m.num_reqs] >= 0).tolist()
        if any(cand):
            spec = cand
    return [i for i in range(m.num_reqs) if not spec[i] and qlens[i] > 1]


def attach(builder, meta, m, num_decode_draft_tokens_cpu):
    """Called at the end of PleShortConvAttentionMetadataBuilder.build: add the checkpoint for the prefill rows."""
    ckb = getattr(builder, "_mbx_ckpt_builder", None)
    if ckb is None or getattr(meta, "num_prefills", 0) == 0:
        return meta
    rows = prefill_rows(m, builder.use_spec_decode, num_decode_draft_tokens_cpu)
    if len(rows) != meta.num_prefills:
        raise RuntimeError(f"mbx ple checkpoint: {len(rows)} prefill rows vs metadata num_prefills {meta.num_prefills}")
    ck = ckb.build(m, rows)
    if ck is None:
        return meta
    from dataclasses import replace

    return replace(meta, mbx_ple_checkpoint=ck)
