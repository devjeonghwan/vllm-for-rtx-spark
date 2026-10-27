"""Compatibility facade for the former ``deep_gemm._C`` extension.

Keeping this module name preserves the existing Python API after moving
operator registration from pybind to ``TORCH_LIBRARY``. It loads the compiled
extension and forwards calls to ``torch.ops.deep_gemm``.
"""

import torch
from pathlib import Path


def _load_extension():
    so_files = list(Path(__file__).parent.glob('_C_extension*.so'))
    assert len(so_files) == 1, (
        f'Expected one _C_extension*.so file, found {len(so_files)}: {so_files}'
    )
    torch.ops.load_library(str(so_files[0]))


_load_extension()
_torch_ops = torch.ops.deep_gemm


def _bind_guarded_ops(*names):
    """Bind ops when all are registered (matches one C++ #if guard group)."""
    present = [name for name in names if hasattr(_torch_ops, name)]
    if not present:
        return
    assert len(present) == len(names), (
        f'Guard group mismatch: {sorted(set(names) - set(present))} missing while '
        f'{present} are registered — the C++ #if guards for these ops have diverged.'
    )
    globals().update({name: getattr(_torch_ops, name) for name in names})


init = _torch_ops.init
set_num_sms = _torch_ops.set_num_sms
get_num_sms = _torch_ops.get_num_sms
set_tc_util = _torch_ops.set_tc_util
get_tc_util = _torch_ops.get_tc_util
set_pdl = _torch_ops.set_pdl
get_pdl = _torch_ops.get_pdl
set_ignore_compile_dims = _torch_ops.set_ignore_compile_dims
get_mk_alignment_for_contiguous_layout = _torch_ops.get_mk_alignment_for_contiguous_layout
get_theoretical_mk_alignment_for_contiguous_layout = _torch_ops.get_theoretical_mk_alignment_for_contiguous_layout
cublaslt_gemm_nt = _torch_ops.cublaslt_gemm_nt
cublaslt_gemm_nn = _torch_ops.cublaslt_gemm_nn
cublaslt_gemm_tn = _torch_ops.cublaslt_gemm_tn
cublaslt_gemm_tt = _torch_ops.cublaslt_gemm_tt


def set_block_size_multiple_of(value):
    if isinstance(value, int):
        return _torch_ops.set_block_size_multiple_of([value])
    return _torch_ops.set_block_size_multiple_of(list(value))


def set_mk_alignment_for_contiguous_layout(value):
    return _torch_ops.set_mk_alignment_for_contiguous_layout(value)


def _unpack_ab_pair(a, b):
    return a[0], a[1], b[0], b[1]


def _unpack_kv(kv):
    return kv[0], kv[1]


def _as_int_list(value):
    """Reject a bare scalar instead of letting int[N] silently broadcast it into a list."""
    return None if value is None else list(value)


def _register_deep_gemm_kernels():
    """Export compatibility wrappers for the registered DeepGEMM kernels."""
    def fp8_fp4_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                        compiled_dims='nk', disable_ue8m0_cast=False, alpha=None):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_fp4_gemm_nt(
            a_tensor, sfa, b_tensor, sfb, d, c, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast, alpha,
        )

    def fp8_fp4_gemm_nn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                        compiled_dims='nk', disable_ue8m0_cast=False, alpha=None):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_fp4_gemm_nn(
            a_tensor, sfa, b_tensor, sfb, d, c, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast, alpha,
        )

    def fp8_fp4_gemm_tn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                        compiled_dims='mn', disable_ue8m0_cast=False, alpha=None):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_fp4_gemm_tn(
            a_tensor, sfa, b_tensor, sfb, d, c, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast, alpha,
        )

    def fp8_fp4_gemm_tt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                        compiled_dims='mn', disable_ue8m0_cast=False, alpha=None):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_fp4_gemm_tt(
            a_tensor, sfa, b_tensor, sfb, d, c, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast, alpha,
        )

    def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None,
                                             compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False,
                                             ensure_zero_padding=True, expected_m_for_psum_layout=None):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a_tensor, sfa, b_tensor, sfb, d, grouped_layout, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding,
            expected_m_for_psum_layout,
        )

    def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None,
                                             compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False,
                                             ensure_zero_padding=True):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.m_grouped_fp8_fp4_gemm_nn_contiguous(
            a_tensor, sfa, b_tensor, sfb, d, grouped_layout, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding,
        )

    def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None,
                                         compiled_dims='nk', disable_ue8m0_cast=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.m_grouped_fp8_fp4_gemm_nt_masked(
            a_tensor, sfa, b_tensor, sfb, d, masked_m, expected_m, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast,
        )

    def k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1, 1, 128),
                                         compiled_dims='mn', use_psum_layout=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.k_grouped_fp8_gemm_tn_contiguous(
            a_tensor, sfa, b_tensor, sfb, d, ks_cpu, grouped_layout, c, list(recipe),
            compiled_dims, use_psum_layout,
        )

    def k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1, 1, 128),
                                         compiled_dims='mn', use_psum_layout=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.k_grouped_fp8_gemm_nt_contiguous(
            a_tensor, sfa, b_tensor, sfb, d, ks_cpu, grouped_layout, c, list(recipe),
            compiled_dims, use_psum_layout,
        )

    def fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, recipe=None, compiled_dims='nk', disable_ue8m0_cast=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_gemm_nt_skip_head_mid(
            a_tensor, sfa, b_tensor, sfb, d, list(head_splits), _as_int_list(recipe), compiled_dims, disable_ue8m0_cast,
        )

    def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128)):
        d_tensor, sfd = d if isinstance(d, (tuple, list)) else (d, None)
        return _torch_ops.fp8_einsum(expr, a[0], a[1], b[0], b[1], d_tensor, c, list(recipe), sfd)

    def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=True,
                           max_seqlen_k=0, logits_dtype=torch.float32, schedule_meta=None):
        q_fp, q_sf = q[0], q[1]
        kv_fp, kv_sf = _unpack_kv(kv)
        return _torch_ops.fp8_fp4_mqa_logits(
            q_fp, q_sf, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end,
            clean_logits, max_seqlen_k, logits_dtype, schedule_meta,
        )

    def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
                                 clean_logits=False, logits_dtype=torch.float32, indices=None):
        q_fp, q_sf = q[0], q[1]
        return _torch_ops.fp8_fp4_paged_mqa_logits(
            q_fp, q_sf, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
            clean_logits, logits_dtype, indices,
        )

    def fp8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=True, max_seqlen_k=0):
        kv_fp, kv_sf = _unpack_kv(kv)
        return _torch_ops.fp8_mqa_logits(q, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, max_seqlen_k)

    def fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
                             clean_logits=False, indices=None):
        return _torch_ops.fp8_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, indices,
        )

    globals().update({
        'fp8_fp4_gemm_nt': fp8_fp4_gemm_nt,
        'fp8_fp4_gemm_nn': fp8_fp4_gemm_nn,
        'fp8_fp4_gemm_tn': fp8_fp4_gemm_tn,
        'fp8_fp4_gemm_tt': fp8_fp4_gemm_tt,
        'fp8_gemm_nt': fp8_fp4_gemm_nt,
        'fp8_gemm_nn': fp8_fp4_gemm_nn,
        'fp8_gemm_tn': fp8_fp4_gemm_tn,
        'fp8_gemm_tt': fp8_fp4_gemm_tt,
        'm_grouped_fp8_fp4_gemm_nt_contiguous': m_grouped_fp8_fp4_gemm_nt_contiguous,
        'm_grouped_fp8_fp4_gemm_nn_contiguous': m_grouped_fp8_fp4_gemm_nn_contiguous,
        'm_grouped_fp8_fp4_gemm_nt_masked': m_grouped_fp8_fp4_gemm_nt_masked,
        'm_grouped_fp8_gemm_nt_contiguous': m_grouped_fp8_fp4_gemm_nt_contiguous,
        'm_grouped_fp8_gemm_nn_contiguous': m_grouped_fp8_fp4_gemm_nn_contiguous,
        'm_grouped_fp8_gemm_nt_masked': m_grouped_fp8_fp4_gemm_nt_masked,
        'k_grouped_fp8_gemm_tn_contiguous': k_grouped_fp8_gemm_tn_contiguous,
        'k_grouped_fp8_gemm_nt_contiguous': k_grouped_fp8_gemm_nt_contiguous,
        'fp8_gemm_nt_skip_head_mid': fp8_gemm_nt_skip_head_mid,
        'fp8_einsum': fp8_einsum,
        'fp8_fp4_mqa_logits': fp8_fp4_mqa_logits,
        'fp8_fp4_paged_mqa_logits': fp8_fp4_paged_mqa_logits,
        'fp8_mqa_logits': fp8_mqa_logits,
        'fp8_paged_mqa_logits': fp8_paged_mqa_logits,
    })

    # BF16 GEMMs
    _bind_guarded_ops(
        'bf16_gemm_nt',
        'bf16_gemm_nn',
        'bf16_gemm_tn',
        'bf16_gemm_tt',
        'm_grouped_bf16_gemm_nt_contiguous',
        'm_grouped_bf16_gemm_nn_contiguous',
        'm_grouped_bf16_gemm_nt_masked',
        'k_grouped_bf16_gemm_tn_contiguous',
    )

    # Einsum, hyperconnection, and attention metadata
    _bind_guarded_ops(
        'einsum',                         # einsum.hpp
        'tf32_hc_prenorm_gemm',           # hyperconnection.hpp
        'get_paged_mqa_logits_metadata',  # attention.hpp
    )

    # Layout helpers
    _bind_guarded_ops(
        'transform_sf_into_required_layout',
        'get_tma_aligned_size',
        'get_mn_major_tma_aligned_tensor',
        'get_mn_major_tma_aligned_packed_ue8m0_tensor',
        'get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor',
    )


_register_deep_gemm_kernels()


def get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads):
    return _torch_ops.get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads)


def get_sparse_mqa_logits_metadata(
    cu_seq_len_k_start,
    cu_seq_len_k_end,
    num_kv_tokens,
    sparse_kv_block_indices,
    qk_dtype,
    sparse_block_kv,
    use_unaligned_ks=False,
):
    return _torch_ops.get_sparse_mqa_logits_metadata(
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        num_kv_tokens,
        sparse_kv_block_indices,
        qk_dtype,
        sparse_block_kv,
        use_unaligned_ks,
    )


def get_paged_sparse_mqa_logits_metadata(
    context_lens, block_table, indices, page_kv, sparse_kv_block_indices, qk_dtype, sparse_block_kv
):
    return _torch_ops.get_paged_sparse_mqa_logits_metadata(
        context_lens, block_table, indices, page_kv, sparse_kv_block_indices, qk_dtype, sparse_block_kv
    )


def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv, use_unaligned_ks=False):
    return _torch_ops.fp8_fp4_sparse_mqa_logits(
        q[0], q[1], kv[0], kv[1], weights, metadata, num_max_sparse_blocks, sparse_block_kv, use_unaligned_ks
    )


def fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv):
    return _torch_ops.fp8_fp4_paged_sparse_mqa_logits(
        q[0], q[1], kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv
    )


def k_grouped_fp4_gemm_nt_contiguous(
    a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1, 1, 32), compiled_dims='mn', use_psum_layout=False
):
    return _torch_ops.k_grouped_fp4_gemm_nt_contiguous(
        a[0], a[1], b[0], b[1], d, ks_cpu, grouped_layout, c, list(recipe), compiled_dims, use_psum_layout
    )


def cublaslt_nvfp4_gemm_nt(a, b, d, c=None):
    return _torch_ops.cublaslt_nvfp4_gemm_nt(a[0], a[1], b[0], b[1], d, c)


def batched_syrk(a, d):
    return _torch_ops.batched_syrk(a, d)


def batched_symm(a, b, d):
    return _torch_ops.batched_symm(a, b, d)


def mega_mhc(
    x,
    residual,
    shifted_prev_mix,
    post_mix,
    comb_res_mix,
    fn,
    mix_scales,
    mix_bases,
    hc_mult,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    num_sinkhorn_iters,
    rmsnorm_weight,
    rmsnorm_eps,
    rmsnorm_scale,
    new_residual,
    new_prev_mix,
    new_post_mix,
    new_comb_res_mix,
    y_bf16=None,
    y_fp8=None,
    y_gemm_sf=None,
    y_routed_sf=None,
    y_shared_sf=None,
    shared_sf_block_m=0,
):
    return _torch_ops.mega_mhc(
        x,
        residual,
        shifted_prev_mix,
        post_mix,
        comb_res_mix,
        fn,
        mix_scales,
        mix_bases,
        hc_mult,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        num_sinkhorn_iters,
        rmsnorm_weight,
        rmsnorm_eps,
        rmsnorm_scale,
        new_residual,
        new_prev_mix,
        new_post_mix,
        new_comb_res_mix,
        y_bf16,
        y_fp8,
        y_gemm_sf,
        y_routed_sf,
        y_shared_sf,
        shared_sf_block_m,
    )


def get_token_alignment_for_mega_moe():
    return _torch_ops.get_token_alignment_for_mega_moe()


def get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type):
    return _torch_ops.get_block_m_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type
    )


def get_token_alignment_for_nvfp4_mega_moe():
    return _torch_ops.get_token_alignment_for_nvfp4_mega_moe()


def get_block_m_for_nvfp4_mega_moe(
    num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk,
):
    return _torch_ops.get_block_m_for_nvfp4_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk
    )


def nvfp4_mega_moe(
    y,
    l1_weights_tuple,
    l2_weights_tuple,
    shared_l1_weights_tuple_opt,
    shared_l2_weights_tuple_opt,
    cumulative_local_expert_recv_stats,
    sym_buffer,
    sym_buffer_ptrs,
    rank_idx,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    activation_clamp_opt,
    fast_math,
    activation_alpha,
    activation_beta,
    l1_alpha=None,
    l2_alpha=None,
    l2_activation_scale=1.0,
    use_x_scales=False,
):
    return _torch_ops.nvfp4_mega_moe(
        y,
        l1_weights_tuple[0],
        l1_weights_tuple[1],
        l2_weights_tuple[0],
        l2_weights_tuple[1],
        shared_l1_weights_tuple_opt[0] if shared_l1_weights_tuple_opt is not None else None,
        shared_l1_weights_tuple_opt[1] if shared_l1_weights_tuple_opt is not None else None,
        shared_l2_weights_tuple_opt[0] if shared_l2_weights_tuple_opt is not None else None,
        shared_l2_weights_tuple_opt[1] if shared_l2_weights_tuple_opt is not None else None,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        sym_buffer_ptrs,
        rank_idx,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        activation_clamp_opt,
        fast_math,
        activation_alpha,
        activation_beta,
        l1_alpha,
        l2_alpha,
        l2_activation_scale,
        use_x_scales,
    )


def fp8_fp4_mega_moe(
    y,
    l1_weights_tuple,
    l2_weights_tuple,
    shared_l1_weights_tuple_opt,
    shared_l2_weights_tuple_opt,
    cumulative_local_expert_recv_stats,
    sym_buffer,
    sym_buffer_ptrs,
    rank_idx,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    recipe,
    activation,
    activation_clamp_opt,
    fast_math,
    activation_alpha,
    activation_beta,
):
    return _torch_ops.fp8_fp4_mega_moe(
        y,
        l1_weights_tuple[0],
        l1_weights_tuple[1],
        l2_weights_tuple[0],
        l2_weights_tuple[1],
        shared_l1_weights_tuple_opt[0] if shared_l1_weights_tuple_opt is not None else None,
        shared_l1_weights_tuple_opt[1] if shared_l1_weights_tuple_opt is not None else None,
        shared_l2_weights_tuple_opt[0] if shared_l2_weights_tuple_opt is not None else None,
        shared_l2_weights_tuple_opt[1] if shared_l2_weights_tuple_opt is not None else None,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        sym_buffer_ptrs,
        rank_idx,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        list(recipe),
        activation,
        activation_clamp_opt,
        fast_math,
        activation_alpha,
        activation_beta,
    )


def bf16_mega_moe(
    y,
    l1_weights,
    l2_weights,
    shared_l1_weights_opt,
    shared_l2_weights_opt,
    cumulative_local_expert_recv_stats,
    sym_buffer,
    sym_buffer_ptrs,
    rank_idx,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    activation,
    activation_clamp_opt,
    fast_math,
    activation_alpha,
    activation_beta,
):
    return _torch_ops.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights_opt,
        shared_l2_weights_opt,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        sym_buffer_ptrs,
        rank_idx,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        activation,
        activation_clamp_opt,
        fast_math,
        activation_alpha,
        activation_beta,
    )


use_deterministic_algorithms = _torch_ops.use_deterministic_algorithms
fp4_gemm_nt = fp8_fp4_gemm_nt
m_grouped_fp4_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp4_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked


def get_bf16_mega_gate_config(num_tokens, hidden, num_routed_experts, num_topk):
    return _torch_ops.get_bf16_mega_gate_config(num_tokens, hidden, num_routed_experts, num_topk)


def bf16_mega_gate(
    x,
    weight,
    num_topk,
    use_shared_as_routed,
    num_shared_experts,
    routed_scaling_factor,
    ep_rank,
    scoring_func='identity',
    mask=None,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    to_physical_map=None,
    logical_count=None,
    unmapped_topk_idx=None,
    force_random=None,
    out=None,
):
    if out is None:
        shape = (x.shape[0], num_topk + (num_shared_experts if use_shared_as_routed else 0))
        out = (
            torch.empty(shape, dtype=torch.int64, device=x.device),
            torch.empty(shape, dtype=torch.float32, device=x.device),
        )
    _torch_ops.bf16_mega_gate(
        x,
        weight,
        num_topk,
        use_shared_as_routed,
        num_shared_experts,
        routed_scaling_factor,
        ep_rank,
        scoring_func,
        mask,
        bias,
        image_bias,
        image_token_mask,
        fix_routing_mask,
        to_physical_map,
        logical_count,
        unmapped_topk_idx,
        force_random,
        *out,
    )
    return tuple(out)


def get_symm_buffer_size_for_mega_moe(
    num_ranks,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    mma_type,
    activation,
    num_shared_experts,
):
    num_bytes, layout_info = _torch_ops.get_symm_buffer_size_for_mega_moe(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        mma_type,
        activation,
        num_shared_experts,
    )

    def slice_input_buffers(buffer):
        return _torch_ops._slice_symm_buffer_for_mega_moe(buffer, layout_info)

    return num_bytes, slice_input_buffers


def get_symm_buffer_size_for_nvfp4_mega_moe(
    num_ranks,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    num_shared_experts=0,
    shared_bf16=False,
):
    num_bytes, layout_info = _torch_ops.get_symm_buffer_size_for_nvfp4_mega_moe(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        shared_bf16,
    )

    def slice_input_buffers(buffer):
        return _torch_ops._slice_symm_buffer_for_nvfp4_mega_moe(buffer, layout_info)

    return num_bytes, slice_input_buffers


Runtime = torch.classes.deep_gemm.Runtime
get_jit = _torch_ops.get_jit


_PUBLIC_API = [
    # Runtime and configuration
    'Runtime', 'get_jit', 'init',
    'set_num_sms', 'get_num_sms',
    'set_tc_util', 'get_tc_util',
    'set_pdl', 'get_pdl',
    'use_deterministic_algorithms',
    'set_ignore_compile_dims',
    'set_block_size_multiple_of',
    'set_mk_alignment_for_contiguous_layout',
    'get_mk_alignment_for_contiguous_layout',
    'get_theoretical_mk_alignment_for_contiguous_layout',
    # cuBLASLt kernels
    'cublaslt_gemm_nt', 'cublaslt_gemm_nn',
    'cublaslt_gemm_tn', 'cublaslt_gemm_tt',
    'cublaslt_nvfp4_gemm_nt',
    'batched_syrk', 'batched_symm',
    # Mega kernels
    'mega_mhc',
    'get_token_alignment_for_mega_moe',
    'get_block_m_for_mega_moe',
    'get_symm_buffer_size_for_mega_moe',
    'fp8_fp4_mega_moe', 'bf16_mega_moe',
    'get_token_alignment_for_nvfp4_mega_moe',
    'get_block_m_for_nvfp4_mega_moe',
    'get_symm_buffer_size_for_nvfp4_mega_moe',
    'nvfp4_mega_moe',
    'get_bf16_mega_gate_config', 'bf16_mega_gate',
    # FP8/FP4 GEMMs
    'fp8_fp4_gemm_nt', 'fp8_fp4_gemm_nn',
    'fp8_fp4_gemm_tn', 'fp8_fp4_gemm_tt',
    'fp8_gemm_nt', 'fp8_gemm_nn',
    'fp8_gemm_tn', 'fp8_gemm_tt',
    'fp4_gemm_nt',
    'm_grouped_fp8_fp4_gemm_nt_contiguous',
    'm_grouped_fp8_fp4_gemm_nn_contiguous',
    'm_grouped_fp8_fp4_gemm_nt_masked',
    'm_grouped_fp8_gemm_nt_contiguous',
    'm_grouped_fp8_gemm_nn_contiguous',
    'm_grouped_fp8_gemm_nt_masked',
    'm_grouped_fp4_gemm_nt_contiguous',
    'm_grouped_fp4_gemm_nt_masked',
    'k_grouped_fp8_gemm_tn_contiguous',
    'k_grouped_fp8_gemm_nt_contiguous',
    'k_grouped_fp4_gemm_nt_contiguous',
    'fp8_gemm_nt_skip_head_mid',
    # BF16 GEMMs
    'bf16_gemm_nt', 'bf16_gemm_nn',
    'bf16_gemm_tn', 'bf16_gemm_tt',
    'm_grouped_bf16_gemm_nt_contiguous',
    'm_grouped_bf16_gemm_nn_contiguous',
    'm_grouped_bf16_gemm_nt_masked',
    'k_grouped_bf16_gemm_tn_contiguous',
    # Einsum
    'einsum', 'fp8_einsum',
    # Attention
    'fp8_fp4_mqa_logits',
    'get_mqa_logits_metadata',
    'get_paged_mqa_logits_metadata',
    'get_sparse_mqa_logits_metadata',
    'get_paged_sparse_mqa_logits_metadata',
    'fp8_fp4_sparse_mqa_logits',
    'fp8_fp4_paged_sparse_mqa_logits',
    'fp8_fp4_paged_mqa_logits',
    'fp8_mqa_logits', 'fp8_paged_mqa_logits',
    # Hyperconnection
    'tf32_hc_prenorm_gemm',
    # Layout
    'transform_sf_into_required_layout',
    'get_tma_aligned_size',
    'get_mn_major_tma_aligned_tensor',
    'get_mn_major_tma_aligned_packed_ue8m0_tensor',
    'get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor',
]

__all__ = _PUBLIC_API
