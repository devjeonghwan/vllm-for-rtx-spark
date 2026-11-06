"""Fused FP8 matmul for vLLM's ChannelWiseTorchFP8ScaledMMLinearKernel on RTX Spark (Windows).

C[m, n] = (A_q[m, :] @ B_q[:, n]) * scale_a[m] * scale_b[n], written in out_dtype.
Replaces the fp32 GEMM + separate scaling passes of the ChannelWise path; the
scales are applied in the GEMM epilogue. Tile sizes are fixed (no autotune) so
the kernel can be captured in CUDA graphs; they are the autotuned winners from
fp8_triton_kernel.py on the RTX Spark N1X for Gemma 4 12B shapes.

Registered as torch.ops.rtxfp8.scaled_mm so torch.compile treats it as one op.
Installed into vLLM by patch_vllm_fp8_fused.py; enabled with VLLM_RTX_FP8_FUSED=1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_scaled_mm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None].to(tl.int64) * stride_am + rk[None, :] * stride_ak
    b = b_ptr + rk[:, None] * stride_bk + rn[None, :].to(tl.int64) * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_left = K - k * BK
        a_t = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b_t = tl.load(b, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a_t, b_t, acc)
        a += BK * stride_ak
        b += BK * stride_bk

    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
    out = (acc * sa[:, None] * sb[None, :]).to(c_ptr.dtype.element_ty)
    c = c_ptr + rm[:, None].to(tl.int64) * stride_cm + rn[None, :] * stride_cn
    tl.store(c, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _tiles(M, K):
    """(BM, BN, BK, num_warps, num_stages) from the RTX Spark autotune results."""
    if K >= 8192:  # down_proj: long K
        return 128, 128, 128, 8, 3
    if M <= 64:
        return 64, 128, 128, 4, 4
    return 128, 128, 64, 8, 4


@torch.library.custom_op("rtxfp8::scaled_mm", mutates_args=())
def scaled_mm(A: torch.Tensor, B: torch.Tensor, scale_a: torch.Tensor,
              scale_b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    M, K = A.shape
    N = B.shape[1]
    sa = scale_a.reshape(-1).float()
    sb = scale_b.reshape(-1).float()
    if sa.numel() == 1:
        sa = sa.expand(M).contiguous()
    if sb.numel() == 1:
        sb = sb.expand(N).contiguous()
    C = torch.empty((M, N), dtype=out_dtype, device=A.device)
    BM, BN, BK, warps, stages = _tiles(M, K)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _fp8_scaled_mm[grid](A, B, sa, sb, C, M, N, K,
                         A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                         C.stride(0), C.stride(1),
                         BM=BM, BN=BN, BK=BK, GROUP_M=8, num_warps=warps, num_stages=stages)
    return C


@scaled_mm.register_fake
def _(A, B, scale_a, scale_b, out_dtype):
    return A.new_empty((A.shape[0], B.shape[1]), dtype=out_dtype)
