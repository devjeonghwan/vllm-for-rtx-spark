// Private adaptation of Strata, copyright (c) 2026 Niko1221 and contributors.
// MIT; see LICENSE.strata. Only HC up projection + gate/mix is included.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>
#include <limits>
namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
// S23 on CUDA (sm_80+; opt-in STRATA_HC_UPMIX=1, see hc_upmix in prefill.cpp): the hyper-connection read's up
// projection (lo16 x w_up^T, BF16 in, FP32 accumulate over K = 320) with gr_mix_r as its epilogue, so `gated`
// (T x 10240 FP32, 336 MB for an 8,192-token chunk) is neither written by the GEMM nor read back by the mix - gr_mix_r
// reads R and `gated` at ~96% of the bandwidth already, so only fewer bytes make the read faster.  A block: 128 tokens
// x 32 columns x the 4 streams (128 output columns), K staged in 32-deep slices (cp.async, double-buffered), 8 warps of
// 32 tokens x 16 columns x 4 streams on mma.m16n8k16 (ldmatrix); a lane then holds its (token, column) pairs for all 4
// streams, and the mix runs in registers in gr_mix_r_kernel's order (x_c = R * rs * w; s = fmaf(x_c, sigm(g_c), s);
// s / 4) with the BF16 / FP16 images.  The epilogue's R rows are asked into L2 at the start.  With cuBLAS's up GEMM
// summing K in the same order (CUDA 13.3 on an RTX 5090 from 33 tokens: gr_upmix_parity) `mixed` is the default pair's
// bits.
constexpr int UC_BM = 128, UC_BD = 32;   // tokens, columns a stream
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800   // cp.async, bf16 mma: the kernel is empty before sm_80
constexpr int UC_KC = 32, UC_LD = UC_KC + 8;   // k a slice, a staged row
__device__ __forceinline__ void uc_pf_l2(const uint16_t* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ void uc_cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void uc_ldm4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void uc_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif
__global__ void __launch_bounds__(256, 2) gr_upmix_cuda_kernel(const uint16_t* __restrict__ lo16,
                                                              const uint16_t* __restrict__ wu,
                                                              const uint16_t* __restrict__ xn,
                                                              uint16_t* __restrict__ mixed,
                                                              int64_t T, int64_t lora_stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) uint16_t sA[2][UC_BM][UC_LD];
    __shared__ __align__(16) uint16_t sB[2][HC * UC_BD][UC_LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    // blocks column-tile fastest: the blocks in flight share a token tile (its lo16 rows stay in L2, the epilogue's R
    // reads run along rows)
    const int64_t t0 = (int64_t) blockIdx.y * UC_BM;
    const int d0 = blockIdx.x * UC_BD;
    // the epilogue's R rows (128 tokens x 4 streams x 128 bytes) asked into L2 now, so they arrive under the GEMM
    for (int i = tid; i < UC_BM * HC; i += 256) {
        const int64_t t = t0 + i / HC;
        if (t < T) uc_pf_l2(xn + t * D + (i % HC) * N + d0);
    }
    auto stage = [&](int kc, int buf) {
        for (int i = tid; i < UC_BM * (UC_KC / 8); i += 256) {
            const int r = i / (UC_KC / 8), q = i % (UC_KC / 8);
            const int64_t t = t0 + r < T ? t0 + r : T - 1;   // the last tile's missing rows read a real row, unused
            uc_cp16(&sA[buf][r][q * 8], lo16 + t * lora_stride + kc * UC_KC + q * 8);
        }
        for (int i = tid; i < HC * UC_BD * (UC_KC / 8); i += 256) {
            const int r = i / (UC_KC / 8), q = i % (UC_KC / 8);
            uc_cp16(&sB[buf][r][q * 8], wu + ((int64_t) (r / UC_BD) * N + d0 + r % UC_BD) * LR + kc * UC_KC + q * 8);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[2][HC][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int c = 0; c < HC; ++c)
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[mt][c][s][i] = 0.0f;
    constexpr int NK = LR / UC_KC;
    stage(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int buf = kc & 1;
        if (kc + 1 < NK) {
            stage(kc + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < UC_KC / 16; ++ks) {
            unsigned a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) uc_ldm4(a[mt], &sA[buf][wm * 32 + mt * 16 + (lane & 15)][ks * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                unsigned b[4];   // b[0..1]: columns 0-7 of this warp's 16, b[2..3]: 8-15 (k 0-7, 8-15 each)
                uc_ldm4(b, &sB[buf][c * UC_BD + wn * 16 + (lane & 7) + ((lane >> 4) << 3)][ks * 16 + ((lane >> 3) & 1) * 8]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    uc_mma(acc[mt][c][0], a[mt], b[0], b[1]);
                    uc_mma(acc[mt][c][1], a[mt], b[2], b[3]);
                }
            }
        }
        __syncthreads();   // every read of `buf` before stage(kc + 2) refills it
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t t = t0 + wm * 32 + mt * 16 + (lane >> 2) + h * 8;
            if (t >= T) continue;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int d = d0 + wn * 16 + s * 8 + (lane & 3) * 2;
                float sm[2] = {0.0f, 0.0f};
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const uint32_t x2 = *reinterpret_cast<const uint32_t*>(xn + t * D + c * N + d);
                    const float x0 = __uint_as_float((x2 & 0xffffu) << 16);
                    const float x1 = __uint_as_float(x2 & 0xffff0000u);
                    // Preserve vLLM's BF16 up-projection output before sigmoid.
                    const float g0 = __bfloat162float(__float2bfloat16_rn(acc[mt][c][s][h * 2 + 0]));
                    const float g1 = __bfloat162float(__float2bfloat16_rn(acc[mt][c][s][h * 2 + 1]));
                    sm[0] = fmaf(sigm(g0), x0, sm[0]);
                    sm[1] = fmaf(sigm(g1), x1, sm[1]);
                }
                sm[0] /= (float) HC;
                sm[1] /= (float) HC;
                *reinterpret_cast<uint32_t*>(mixed + t * N + d) =
                    uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sm[0]))) |
                    (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sm[1]))) << 16);
            }
        }
#endif
}

bool overlaps(uintptr_t a, size_t an, uintptr_t b, size_t bn) { return a < b + bn && b < a + an; }
}
extern "C" __declspec(dllexport) int rtx_hc_upmix_abi() { return 2; }
extern "C" __declspec(dllexport) int rtx_hc_upmix_bf16(const uint16_t* lora, const uint16_t* weight,
                                                     const uint16_t* xn, uint16_t* output,
                                                     int64_t rows, int64_t lora_stride, void* stream) {
    if (rows < 1 || rows > 8192 || (lora_stride != 320 && lora_stride != 336) || !lora || !weight || !xn || !output) return -1;
    const uintptr_t a = reinterpret_cast<uintptr_t>(lora), w = reinterpret_cast<uintptr_t>(weight),
                    x = reinterpret_cast<uintptr_t>(xn), y = reinterpret_cast<uintptr_t>(output);
    if ((a | w | x | y) % 16) return -2;
    const size_t an = size_t((rows - 1) * lora_stride + LR) * 2, wn = size_t(D) * LR * 2,
                 xnbytes = size_t(rows) * D * 2, yn = size_t(rows) * N * 2;
    if (a > UINTPTR_MAX - an || w > UINTPTR_MAX - wn || x > UINTPTR_MAX - xnbytes || y > UINTPTR_MAX - yn) return -3;
    if (overlaps(y, yn, a, an) || overlaps(y, yn, w, wn) || overlaps(y, yn, x, xnbytes)) return -4;
    // stream==0 is CUDA's valid default stream; non-default handles pass through unchanged.
    gr_upmix_cuda_kernel<<<dim3(N / UC_BD, unsigned((rows + UC_BM - 1) / UC_BM)), 256, 0,
                          static_cast<cudaStream_t>(stream)>>>(lora, weight, xn, output, rows, lora_stride);
    return int(cudaGetLastError());
}
