#pragma once

#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::ptx {

struct SM100_MMA_NVF4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a, uint64_t const& desc_b,
        uint32_t const& tmem_c, uint32_t const& scale_c,
        uint64_t const& desc, uint32_t const& tmem_sfa, uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;\n\t"
#else
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p;\n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)),
               "r"(scale_c), "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

} // namespace deep_gemm::ptx
