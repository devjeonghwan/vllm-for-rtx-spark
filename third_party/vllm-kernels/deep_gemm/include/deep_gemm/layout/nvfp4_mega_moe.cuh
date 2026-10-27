#pragma once

#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

struct NVFP4MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // Shared inputs have independent storage and precision.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           combine_token_buffer;

    // Optional FP32 per-token outer scales of the routed L1 input
    Buffer input_x_scales_buffer,
           l1_x_scales_buffer;

    CUTLASS_HOST_DEVICE
    NVFP4MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const uint32_t& num_shared_experts = 0,
                  const bool& shared_bf16 = false) {
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const bool shared_with_sf = not shared_bf16;
        const auto num_max_shared_sf_tokens = shared_with_sf ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        const uint32_t shared_elem_bytes = shared_with_sf ? 1 : 2;
        const auto input_token_layout = layout::Data(hidden / 2);
        const auto shared_input_token_layout = layout::Data(hidden * shared_elem_bytes);
        const auto bf16_token_layout = layout::Data(hidden * 2);
        const auto intermediate_token_layout = layout::Data(intermediate_hidden / 2);
        const auto shared_intermediate_token_layout = layout::Data(shared_intermediate_hidden * shared_elem_bytes);
        const auto input_sf_layout = layout::Data(hidden / 16, false);
        const auto shared_input_sf_layout = layout::Data(shared_with_sf ? hidden / 32 : 0, false);
        const auto intermediate_sf_layout = layout::Data(intermediate_hidden / 16, false);
        const auto shared_intermediate_sf_layout = layout::Data(shared_with_sf ? shared_intermediate_hidden / 32 : 0, false);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto x_scales_layout = layout::Data(sizeof(float), false);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            workspace.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            input_sf_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        // Shared expert buffers
        shared_l1_token_buffer = Buffer(
            shared_input_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            input_topk_weights_buffer.get_end_ptr());
        shared_l1_sf_buffer = Buffer(
            shared_input_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l1_token_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            shared_l1_sf_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l2_token_buffer.get_end_ptr());

        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                shared_l2_sf_buffer.get_end_ptr() :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());

        // NOTES: top-k weights stay on the source rank and are applied in combine
        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_sf_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            l2_sf_buffer.get_end_ptr());

        // Optional per-token input scales
        input_x_scales_buffer = Buffer(
            x_scales_layout, 1, num_max_tokens_per_rank,
            combine_token_buffer.get_end_ptr());
        l1_x_scales_buffer = Buffer(
            x_scales_layout, 1, num_ring_tokens,
            input_x_scales_buffer.get_end_ptr());
    }

    CUTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(l1_x_scales_buffer.get_end_ptr())
               - reinterpret_cast<uint8_t*>(workspace.signals);
    }
};

} // namespace deep_gemm::layout
