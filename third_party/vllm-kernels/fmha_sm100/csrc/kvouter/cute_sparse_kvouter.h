/*
 * Copyright (c) 2026 Fireworks AI
 * SPDX-License-Identifier: Apache-2.0
 *
 * C++ backend for the cuteDSL KV-outer sparse-attention pipeline. Loads the
 * AOT-exported (CuTe ABI) kernels and drives the full pipeline -- index build,
 * load-balance scheduler, KV-outer forward, and log-sum-exp combine -- doing the
 * torch glue in ATen so the per-call Python / cuteDSL dispatch overhead is gone.
 *
 * The op is config-agnostic: every dimension/config (heads, page/block size,
 * topk, dtypes, num_splits, ...) is supplied to `sparse_kvouter_init`; nothing is
 * hard-coded. The hot path NEVER JIT-compiles -- a request whose offsets variant
 * is not pre-registered is a fatal error.
 */
#pragma once

#include <optional>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace fmha_sm100 {

// Register an AOT-exported KV-outer pipeline and return an opaque handle id that
// `sparse_kvouter_attn` takes as its first argument. Called once per deployment
// config by the Python layer (cpp_backend.py) after it AOT-exports the kernels.
//
// The kernels bake in Hkv / topk / head_dim / dtypes / num_splits, so this handle
// is only valid for requests with the matching config.
//
// Params (the three arrays are PARALLEL — entry i describes one kernel):
//   slots         Logical kernel names, each one of:
//                   "init", "count", "offsets:serial", "offsets:parallel",
//                   "scatter", "scheduler", "forward", "combine".
//                 Both offsets variants MUST be registered; the op picks one per
//                 request from `num_block_slots` vs `offsets_threshold`.
//   object_paths  Filesystem path to each kernel's AOT `.o` (read as bytes and
//                 loaded in-memory via CuteDSLRT_Module_Create_From_Bytes).
//   prefixes      The function-prefix symbol each `.o` was exported with
//                 (passed to CuteDSLRT_Module_Get_Function).
//   runtime_libs  Shared libraries the CuTe DSL runtime needs to JIT-link the
//                 object in memory (typically just libcute_dsl_runtime.so, from
//                 cute.runtime.find_runtime_libraries(enable_tvm_ffi=False)).
//   topk          Top-k width per (query, kv-head); must equal selected.size(2).
//   block_size    Sparse KV block size in tokens (e.g. 128).
//   page_size     Paged-cache page size in tokens (64 or 128). ratio = block/page.
//   num_splits    Load-balance scheduler grid size = device SM count (the forward
//                 grid + nqps host math both use this).
//   return_lse    If true, the combine kernel was exported to also write LSE and
//                 `sparse_kvouter_attn` returns it; if false, lse is undefined.
//   partial_dtype_code  dtype of O_partial / m / l scratch. Code: 0=bf16,1=fp16,2=fp32.
//   out_dtype_code      dtype of the returned output O. Same code mapping.
//   offsets_threshold   num_block_slots strictly greater than this selects the
//                 "offsets:parallel" kernel, else "offsets:serial" (the only
//                 request-variable kernel choice). Both variants are always
//                 registered and compute the identical prefix sum, so this is a
//                 perf-only heuristic (serial wins for small num_block_slots,
//                 parallel for large) -- not a correctness or recompilation knob.
//                 Pass the Python _COUNT_TO_OFFSETS_PARALLEL_THRESHOLD to mirror
//                 the validated reference behavior.
// Returns: an int handle id (index into the process-global handle registry).
int64_t sparse_kvouter_init(
    std::vector<std::string> slots,
    std::vector<std::string> object_paths,
    std::vector<std::string> prefixes,
    std::vector<std::string> runtime_libs,
    int64_t topk,
    int64_t block_size,
    int64_t page_size,
    int64_t num_splits,
    bool return_lse,
    int64_t partial_dtype_code,
    int64_t out_dtype_code,
    int64_t offsets_threshold);

// Run the full pipeline (index build -> scheduler -> forward -> combine) for one
// request on q's device/current stream. All ATen glue runs here; the precompiled
// kernels are invoked via the CuTe ABI. NEVER recompiles — a request whose offsets
// variant isn't registered is a fatal error.
//
// Params:
//   handle        Handle id returned by `sparse_kvouter_init` for this config.
//   q             Queries, token-major [Tq, Hq, D], q.dtype bf16/fp16/fp8
//                 (fp8 is passed through as raw bytes; head_dim D must match the
//                 compiled kernels).
//   k_cache       Paged key cache [num_pages, Hkv, page_size, D], same dtype as q.
//                 Hkv (= k_cache.size(1)) must equal selected.size(1).
//   v_cache       Paged value cache [num_pages, Hkv, page_size, D], same dtype as q.
//   selected      Per-query selected KV block ids [Tq, Hkv, topK] int32 (-1 pads
//                 unused ranks). topK must equal the configured topk.
//   block_tables  Paged block table [B, max_blocks] int32 (logical->physical page
//                 ids per sequence). Must have >= B rows (B = cu_seqlens_q.numel()-1).
//   cu_seqlens_q  Cumulative query lengths [B+1] (cast to int64 here); use [0, Tq]
//                 for a single sequence.
//   used_kv_lens  Real per-sequence KV length Lk_b [B] int32 (cast here); drives the
//                 in-kernel causal/padding mask. Length must equal B.
//   softmax_scale QK softmax scale (typically 1/sqrt(D)).
//   replicas      Adaptive index-counter replica count for this request (a power of
//   out_opt       Optional preallocated [Tq, Hq, D] output (out_dtype, contiguous, on
//                 q's device); written in place and returned as `o`.
//                 two in [16, 128]); selects the matching pre-registered
//                 init:r<R>/count:r<R>/reduce:r<R>/scatter:r<R> kernels and sizes the
//                 3D count buffer [Hkv, num_block_slots, R]. Computed by the Python
//                 layer (mirrors _adaptive_replicas) and is request-variable.
// Returns: tuple (o, lse):
//   o    Attention output, token-major [Tq, Hq, D] in out_dtype.
//   lse  Log-sum-exp [Hq, Tq] fp32 if the handle was inited with return_lse=true,
//        otherwise an undefined tensor (maps to None on the Python side).
std::tuple<at::Tensor, at::Tensor> sparse_kvouter_attn(
    int64_t handle,
    at::Tensor q,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor selected,
    at::Tensor block_tables,
    at::Tensor cu_seqlens_q,
    at::Tensor used_kv_lens,
    double softmax_scale,
    int64_t replicas,
    std::optional<at::Tensor> out_opt);

}  // namespace fmha_sm100
