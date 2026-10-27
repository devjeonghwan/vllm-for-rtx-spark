"""Standalone NVFP4 MegaMoE API; the upstream MegaMoE interfaces are unchanged."""
import types
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .. import _C
from ..utils.math import align


class NVFP4SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 base: Optional['NVFP4SymmBuffer'] = None,
                 shared_dtype: torch.dtype = torch.float8_e4m3fn):
        # Align token count
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_nvfp4_mega_moe())

        # Init
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = 'nvfp4'
        self.activation = 'swiglu'
        self.shared_dtype = shared_dtype
        assert self.shared_dtype in (torch.bfloat16, torch.float8_e4m3fn)

        # Allocate or reuse a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_nvfp4_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            num_shared_experts, self.shared_dtype == torch.bfloat16
        )
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf,
         self.x_scales) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.x_scales = None


def nvfp4_mega_moe(y: torch.Tensor,
                  l1_weights: Tuple[torch.Tensor, torch.Tensor],
                  l2_weights: Tuple[torch.Tensor, torch.Tensor],
                  sym_buffer: NVFP4SymmBuffer,
                  shared_l1_weights: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                  shared_l2_weights: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True,
                  activation_alpha: float = 1.0,
                  activation_beta: float = 0.0,
                  l1_alpha: Optional[torch.Tensor] = None,
                  l2_alpha: Optional[torch.Tensor] = None,
                  l2_activation_scale: float = 1.0,
                  use_x_scales: bool = False):
    """Fused NVFP4 routed experts and MXFP8 or BF16 shared experts (SM100/SM103).

    Routed inputs/weights are packed E2M1 with one E4M3 scale per 16 values;
    four scale bytes are packed into each int32. Shared inputs/weights use
    E4M3 with per-32 UE8M0 scales. Use ``transform_weights_for_mega_moe`` on
    both weight pairs and populate the separate shared input views.
    For BF16 shared experts, create the buffer with ``shared_dtype=torch.bfloat16``
    and pass transformed BF16 weight tensors directly, without scale tensors.

    Optional CUDA float32 vectors ``l1_alpha`` and ``l2_alpha`` (one entry per
    local expert) multiply GEMM accumulators before BF16 rounding. They contain
    the products of activation and weight global dequantization scales.
    ``l2_activation_scale`` is the positive global dequantization scale used
    when quantizing routed SwiGLU outputs; include it in ``l2_alpha``.
    Defaults use unit global scales. The routed L2 input is the unweighted
    SwiGLU output, so calibrate ``l2_activation_scale`` on it (as for a
    checkpoint's down-projection input scale). Combine multiplies each routed
    BF16 output by its top-k weight and accumulates in FP32; the shared expert
    output has unit weight.
    With ``use_x_scales``, ``sym_buffer.x_scales`` holds one FP32 dequantization
    scale per routed input token; it multiplies that token's L1 accumulator
    together with ``l1_alpha``, before BF16 rounding and the activation.
    """
    assert isinstance(sym_buffer, NVFP4SymmBuffer), 'NVFP4 requires NVFP4SymmBuffer'
    assert (shared_l1_weights is not None) == (sym_buffer.num_shared_experts > 0)
    assert (shared_l2_weights is not None) == (sym_buffer.num_shared_experts > 0)
    if isinstance(shared_l1_weights, torch.Tensor):
        assert sym_buffer.shared_dtype == torch.bfloat16
        assert isinstance(shared_l2_weights, torch.Tensor)
        shared_l1_weights = (shared_l1_weights, None)
        shared_l2_weights = (shared_l2_weights, None)
    if shared_l2_weights is not None:
        assert shared_l1_weights[0].dtype == shared_l2_weights[0].dtype == sym_buffer.shared_dtype
        assert shared_l2_weights[0].shape == (sym_buffer.hidden,
                                             sym_buffer.intermediate_hidden * sym_buffer.num_shared_experts)
    _C.nvfp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        activation_clamp, fast_math,
        activation_alpha, activation_beta,
        l1_alpha, l2_alpha, l2_activation_scale, use_x_scales
    )
