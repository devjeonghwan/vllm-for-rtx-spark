# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Michael Valente. Original device functions; see NOTICE.
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
