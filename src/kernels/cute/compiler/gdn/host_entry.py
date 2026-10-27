"""Typed C-ABI forwarding only; original GDN host/device methods stay unchanged."""
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


@cute.jit
def c_abi_entry(
    kernel: cutlass.Constexpr,
    g_q: cute.Tensor, g_k: cute.Tensor, g_v: cute.Tensor, g_o: cute.Tensor,
    g_alpha: cute.Tensor, g_beta: cute.Tensor, g_state: cute.Tensor,
    g_init_state: cute.Tensor, g_state_indices: cute.Tensor,
    g_state_checkpoints: cute.Tensor, checkpoint_cu_starts: cute.Tensor,
    g_tensormaps: cute.Tensor, cu_seqlens: cute.Tensor,
    scale: cutlass.Float32,
    num_q_heads: cutlass.Int32, num_k_heads: cutlass.Int32,
    num_v_heads: cutlass.Int32, num_sab_heads: cutlass.Int32,
    num_seqs: cutlass.Int32, total_checkpoints: cutlass.Int32,
    checkpoint_every_n_tokens: cutlass.Int32, grid_x: cutlass.Int32, stream: cuda.CUstream,
):
    kernel(g_q, g_k, g_v, g_o, g_alpha, g_beta, g_state, g_init_state,
           g_state_indices, g_state_checkpoints, checkpoint_cu_starts,
           g_tensormaps, cu_seqlens, scale, num_q_heads, num_k_heads,
           num_v_heads, num_sab_heads, num_seqs, total_checkpoints,
           checkpoint_every_n_tokens, grid_x, stream)
