// Namespace differs only to coexist with original/previous component ops.
STABLE_TORCH_LIBRARY_FRAGMENT(mbx_marlin_csf_native, m) {
  m.def(R"SCHEMA(moe_wna16_marlin_gemm_csf(Tensor! a, Tensor? c_or_none, Tensor! b_q_weight, Tensor? b_bias_or_none, Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor! workspace, Tensor sorted_token_ids, Tensor! expert_ids, Tensor! num_tokens_past_padded, Tensor! topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, int b_type_id, int size_m, int size_n, int size_k, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float, int thread_k, int thread_n, int blocks_per_sm, Tensor? csf_base, Tensor? csf_nib, Tensor? csf_raw) -> Tensor)SCHEMA");
}
STABLE_TORCH_LIBRARY_IMPL(mbx_marlin_csf_native, CUDA, m) {
  m.impl("moe_wna16_marlin_gemm_csf", TORCH_BOX(&moe_wna16_marlin_gemm_csf));
}
