/*
 * Copyright (c) 2026 Fireworks AI
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of the cuteDSL KV-outer sparse-attention C++ backend. See
 * cute_sparse_kvouter.h. The CuTe ABI for an exported kernel argument is:
 *   tensor : { void* data; int32_t shapes[rank]; int64_t strides[rank-1]; }
 *            (leading dim is the contiguous last dim, excluded from strides)
 *   scalar : pointer to the int32/int64/float value
 *   stream : pointer to a cudaStream_t
 *   trailing &ret (int32); num_args includes the ret slot.
 * The argument layout matches the CuTe AOT runtime ABI.
 */
#include "cute_sparse_kvouter.h"

#include "CuteDSLRuntime.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace fmha_sm100 {

namespace {

// Thread count of the parallel count->offsets kernel (mirrors
// _CountToOffsetsParallelKernel._NUM_THREADS in build_kvouter_index.py). The kernel chunks the
// [0, nbs] compact-j axis across this many threads, so chunk_size = ceil((nbs + 1) / this).
constexpr int64_t kOffsetsParallelThreads = 256;

void cute_check(CuteDSLRT_Error_t e, const char* what) {
  TORCH_CHECK(e == CuteDSLRT_Success, "cute runtime ", what, " failed: ",
              CuteDSLRT_GetErrorName(e), " (", CuteDSLRT_GetErrorString(e), ")");
}

std::vector<unsigned char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(f.good(), "cannot open AOT object file: ", path);
  std::streamsize sz = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<unsigned char> buf(static_cast<size_t>(sz));
  TORCH_CHECK(f.read(reinterpret_cast<char*>(buf.data()), sz).good(),
              "failed reading AOT object file: ", path);
  return buf;
}

at::ScalarType dtype_from_code(int64_t code) {
  switch (code) {
    case 0: return at::kBFloat16;
    case 1: return at::kHalf;
    case 2: return at::kFloat;
    default: TORCH_CHECK(false, "unsupported dtype code ", code);
  }
}

// A single loaded AOT kernel (module + function handle).
struct CuteKernel {
  CuteDSLRT_Module_t* module = nullptr;
  CuteDSLRT_Function_t* func = nullptr;
  ~CuteKernel() {
    if (module) CuteDSLRT_Module_Destroy(module);
  }
};

// Builds the packed argument array for CuteDSLRT_Function_Run. Backing storage is
// kept in a node-stable list so the void* pointers remain valid until run().
class ArgList {
 public:
  void add_tensor(const at::Tensor& t) {
    const int r = static_cast<int>(t.dim());
    const size_t shapes_off = sizeof(void*);
    const size_t shapes_end = shapes_off + sizeof(int32_t) * static_cast<size_t>(r);
    const size_t strides_off = (shapes_end + 7u) & ~size_t(7);  // 8-byte align
    const int n_strides = r > 0 ? r - 1 : 0;  // leading (last) dim excluded
    const size_t total = strides_off + sizeof(int64_t) * static_cast<size_t>(n_strides);
    storage_.emplace_back(total, char(0));
    char* p = storage_.back().data();
    *reinterpret_cast<void**>(p) = t.data_ptr();
    auto* shapes = reinterpret_cast<int32_t*>(p + shapes_off);
    for (int i = 0; i < r; ++i) shapes[i] = static_cast<int32_t>(t.size(i));
    auto* strides = reinterpret_cast<int64_t*>(p + strides_off);
    int k = 0;
    for (int i = 0; i < r; ++i) {
      if (i == r - 1) continue;  // leading dim is the contiguous last dim
      strides[k++] = static_cast<int64_t>(t.stride(i));
    }
    ptrs_.push_back(p);
  }

  void add_i32(int32_t v) { add_scalar(&v, sizeof(v)); }
  void add_i64(int64_t v) { add_scalar(&v, sizeof(v)); }
  void add_f32(float v) { add_scalar(&v, sizeof(v)); }
  void add_stream(cudaStream_t s) { add_scalar(&s, sizeof(s)); }

  void run(CuteDSLRT_Function_t* func) {
    int32_t ret = 0;
    add_scalar(&ret, sizeof(ret));  // trailing &ret slot
    cute_check(CuteDSLRT_Function_Run(func, ptrs_.data(), ptrs_.size()), "Function_Run");
  }

 private:
  void add_scalar(const void* v, size_t n) {
    storage_.emplace_back(n, char(0));
    std::memcpy(storage_.back().data(), v, n);
    ptrs_.push_back(storage_.back().data());
  }

  std::list<std::vector<char>> storage_;  // node-stable addresses
  std::vector<void*> ptrs_;
};

struct KvouterHandle {
  std::unordered_map<std::string, std::unique_ptr<CuteKernel>> kernels;
  int64_t topk = 0;
  int64_t block_size = 0;
  int64_t page_size = 0;
  int64_t num_splits = 0;
  bool return_lse = false;
  at::ScalarType partial_dtype = at::kBFloat16;
  at::ScalarType out_dtype = at::kBFloat16;
  int64_t offsets_threshold = 128;

  CuteDSLRT_Function_t* fn(const std::string& slot) const {
    auto it = kernels.find(slot);
    TORCH_CHECK(it != kernels.end(),
                "fmha_sm100 KV-outer: kernel slot '", slot,
                "' not registered -- this request would require recompilation "
                "(fatal; pre-export all reachable kernels at init)");
    return it->second->func;
  }
};

std::mutex g_mutex;

// Intentionally leaked. The loaded CuTe modules are process-lifetime (handles
// are only ever added, never erased). If this container were destroyed at
// static teardown, each ~CuteKernel would call CuteDSLRT_Module_Destroy ->
// cuModuleUnload after CUDA/torch atexit handlers have already torn down the
// CUDA context, producing a spurious "error unloading compiled function" line
// per loaded kernel. The driver reclaims all GPU resources on process exit, so
// we never run those destructors during shutdown.
std::vector<std::unique_ptr<KvouterHandle>>& g_handles() {
  static auto* handles = new std::vector<std::unique_ptr<KvouterHandle>>();
  return *handles;
}

KvouterHandle& get_handle(int64_t h) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto& handles = g_handles();
  TORCH_CHECK(h >= 0 && static_cast<size_t>(h) < handles.size() && handles[h],
              "invalid fmha_sm100 KV-outer handle ", h);
  return *handles[h];
}

}  // namespace

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
    int64_t offsets_threshold) {
  TORCH_CHECK(slots.size() == object_paths.size() && slots.size() == prefixes.size(),
              "slots/object_paths/prefixes must be parallel arrays");

  std::vector<const char*> libs;
  libs.reserve(runtime_libs.size());
  for (const auto& s : runtime_libs) libs.push_back(s.c_str());

  auto handle = std::make_unique<KvouterHandle>();
  handle->topk = topk;
  handle->block_size = block_size;
  handle->page_size = page_size;
  handle->num_splits = num_splits;
  handle->return_lse = return_lse;
  handle->partial_dtype = dtype_from_code(partial_dtype_code);
  handle->out_dtype = dtype_from_code(out_dtype_code);
  handle->offsets_threshold = offsets_threshold;

  for (size_t i = 0; i < slots.size(); ++i) {
    auto bytes = read_file(object_paths[i]);
    auto kernel = std::make_unique<CuteKernel>();
    cute_check(
        CuteDSLRT_Module_Create_From_Bytes(
            &kernel->module, bytes.data(), bytes.size(),
            libs.empty() ? nullptr : libs.data(), libs.size()),
        ("Module_Create_From_Bytes[" + slots[i] + "]").c_str());
    cute_check(
        CuteDSLRT_Module_Get_Function(&kernel->func, kernel->module, prefixes[i].c_str()),
        ("Module_Get_Function[" + slots[i] + "]").c_str());
    handle->kernels[slots[i]] = std::move(kernel);
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  auto& handles = g_handles();
  handles.push_back(std::move(handle));
  return static_cast<int64_t>(handles.size() - 1);
}

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
    std::optional<at::Tensor> out_opt) {
  const KvouterHandle& H = get_handle(handle);
  const auto device = q.device();
  // Pin all allocations + kernel launches to q's device so we never run on the wrong
  // GPU when the caller's current device differs from the tensors' device.
  const c10::cuda::CUDAGuard device_guard(device);
  const auto i32 = at::TensorOptions().dtype(at::kInt).device(device);
  const auto i64 = at::TensorOptions().dtype(at::kLong).device(device);
  const auto f32 = at::TensorOptions().dtype(at::kFloat).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

  // ---- derive dimensions / scalars (config-agnostic) ----
  const int64_t topk = H.topk;
  const int64_t block_size = H.block_size;
  const int64_t page_size = H.page_size;
  const int64_t ratio = block_size / page_size;
  const int64_t tq = q.size(0);
  const int64_t hq = q.size(1);
  const int64_t d = q.size(2);
  const int64_t hkv = k_cache.size(1);
  const int64_t qhead = hq / hkv;
  const int64_t h_idx = selected.size(1);
  const int64_t cap = tq * h_idx * topk;
  const int64_t n_batches = cu_seqlens_q.numel() - 1;
  const int64_t block_table_cols = block_tables.size(1);

  // Validate inputs against the kernels' compiled config (the .o files bake in hkv,
  // topk, head_dim). A mismatch would index out of bounds or score the wrong heads,
  // so reject it here (mirrors the Python index builder's asserts). Shape scalars
  // like tq / batch / num_block_slots are genuine runtime args and need no check.
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4 && selected.dim() == 3,
              "fmha_sm100 KV-outer: expected q[Tq,Hq,D], k/v_cache[pages,Hkv,page,D], selected[Tq,Hkv,topK]");
  TORCH_CHECK(h_idx == hkv,
              "fmha_sm100 KV-outer: selected head dim (", h_idx, ") must equal k_cache Hkv (", hkv,
              "); kernels are compiled for a fixed Hkv");
  TORCH_CHECK(hq % hkv == 0, "fmha_sm100 KV-outer: Hq (", hq, ") must be a multiple of Hkv (", hkv, ")");
  TORCH_CHECK(topk == selected.size(2),
              "fmha_sm100 KV-outer: selected topK (", selected.size(2), ") must equal the configured topk (", topk, ")");
  TORCH_CHECK(v_cache.size(1) == hkv && v_cache.size(3) == d && k_cache.size(3) == d,
              "fmha_sm100 KV-outer: k/v_cache must share q's head_dim and Hkv");
  TORCH_CHECK(block_size % page_size == 0 && block_tables.dim() == 2,
              "fmha_sm100 KV-outer: block_size must be a multiple of page_size and block_tables 2-D");
  // The index kernels index block_tables[seq_id] / used_kv_lens[seq_id] for seq_id in
  // [0, n_batches), so both must cover every sequence (mirrors the Python builder).
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && n_batches >= 1,
              "fmha_sm100 KV-outer: cu_seqlens_q must be 1-D [B+1] (got numel ", cu_seqlens_q.numel(), ")");
  TORCH_CHECK(block_tables.size(0) >= n_batches,
              "fmha_sm100 KV-outer: block_tables must have >= n_batches=", n_batches, " rows, got ",
              block_tables.size(0));
  TORCH_CHECK(used_kv_lens.numel() == n_batches,
              "fmha_sm100 KV-outer: used_kv_lens length (", used_kv_lens.numel(), ") must equal n_batches (",
              n_batches, ")");
  const int64_t msb = std::max<int64_t>(1, block_tables.size(1) / ratio);
  const int64_t num_block_slots = (n_batches == 1) ? msb : (n_batches * msb);
  const bool parallel = num_block_slots > H.offsets_threshold;
  // +1 so the fused-compaction plateau loop covers the compact-j endpoint nbs (sel_offsets[nbs]);
  // ceil((nbs + 1) / kOffsetsParallelThreads).
  const int64_t chunk_size = (num_block_slots + 1 + kOffsetsParallelThreads - 1) / kOffsetsParallelThreads;
  const int64_t seg = tq * topk * qhead;
  const int64_t grid_size = H.num_splits;

  auto cuq = cu_seqlens_q.to(at::kLong).contiguous();
  auto sk = used_kv_lens.to(at::kInt).contiguous();
  auto sel = selected.contiguous();
  auto bt = block_tables.to(at::kInt).contiguous();

  // ============================= index build ============================= //
  // count is privatized across `replicas` per-slot counters (3D) to cut atomic
  // contention; reduce collapses them into count_total (2D), then offsets prefix-sums
  // it. The per-replica index kernels are selected by the replicas-keyed slot name.
  TORCH_CHECK(replicas >= 1, "fmha_sm100 KV-outer: replicas must be >= 1, got ", replicas);
  const std::string rsfx = ":r" + std::to_string(replicas);
  auto count = at::empty({hkv, num_block_slots, replicas}, i32);  // per-replica counters
  auto count_total = at::empty({hkv, num_block_slots}, i32);      // per-slot total (reduced)
  auto edge_local = at::empty({tq * h_idx * topk}, i32);
  auto slot = at::empty({hkv, num_block_slots * ratio}, i64);
  auto offs = at::empty({hkv, num_block_slots + 1}, i32);
  // Compact selected-slot index, produced FUSED inside the offsets kernel (no separate launch):
  // sel_slots[j] = j-th selected slot, sel_offsets = compact CSR plateaued at the head total,
  // num_sel = selected slots per head. The kernel scatters only sel_slots[0, num_sel); the tail
  // [num_sel, nbs) is left UNINITIALIZED (at::empty, no -1 fill) because the scheduler + forward
  // iterate only [0, num_sel) (bounded by num_sel) -- avoids a per-call fill-kernel launch.
  auto sel_slots = at::empty({hkv, num_block_slots}, i32);
  auto sel_offsets = at::empty({hkv, num_block_slots + 1}, i32);
  auto num_sel = at::empty({hkv}, i32);
  auto idx_ranks = at::empty({hkv, tq * topk, 2}, i32);
  auto inv = at::empty({hkv, tq, topk}, i32);
  const int64_t num_units = hkv * num_block_slots;

  {
    ArgList a;
    a.add_tensor(sel);
    a.add_tensor(bt);
    a.add_tensor(count);
    a.add_tensor(slot);
    a.add_i32(static_cast<int32_t>(num_block_slots));
    a.add_i32(static_cast<int32_t>(msb));
    a.add_i32(static_cast<int32_t>(block_table_cols));
    a.add_stream(stream);
    a.run(H.fn("init" + rsfx));
  }
  {
    ArgList a;
    a.add_tensor(sel);
    a.add_tensor(cuq);
    a.add_tensor(sk);
    a.add_tensor(slot);
    a.add_tensor(count);
    a.add_tensor(edge_local);
    a.add_i32(static_cast<int32_t>(cap));
    a.add_i32(static_cast<int32_t>(msb));
    a.add_i32(static_cast<int32_t>(n_batches));
    a.add_stream(stream);
    a.run(H.fn("count" + rsfx));
  }
  {
    // reduce R replica counters -> count_total (+ overwrite count with replica prefix base)
    ArgList a;
    a.add_tensor(count);
    a.add_tensor(count_total);
    a.add_i32(static_cast<int32_t>(num_units));
    a.add_i32(static_cast<int32_t>(num_block_slots));
    a.add_stream(stream);
    a.run(H.fn("reduce" + rsfx));
  }
  {
    // The offsets kernel does the dense prefix sum AND fuses the selected-slot compaction
    // (sel_slots/sel_offsets/num_sel) in the same launch -- no separate compaction kernel.
    ArgList a;
    a.add_tensor(count_total);
    a.add_tensor(offs);
    a.add_tensor(sel_slots);
    a.add_tensor(sel_offsets);
    a.add_tensor(num_sel);
    a.add_i32(static_cast<int32_t>(num_block_slots));
    if (parallel) a.add_i32(static_cast<int32_t>(chunk_size));
    a.add_stream(stream);
    a.run(H.fn(parallel ? "offsets:parallel" : "offsets:serial"));
  }
  {
    ArgList a;
    a.add_tensor(edge_local);
    a.add_tensor(sel);
    a.add_tensor(cuq);
    a.add_tensor(sk);
    a.add_tensor(slot);
    a.add_tensor(offs);
    a.add_tensor(count);  // replica exclusive-prefix base
    a.add_tensor(idx_ranks);
    a.add_tensor(inv);
    a.add_i32(static_cast<int32_t>(cap));
    a.add_i32(static_cast<int32_t>(msb));
    a.add_i32(static_cast<int32_t>(n_batches));
    a.add_stream(stream);
    a.run(H.fn("scatter" + rsfx));
  }

  // ============================== scheduler ============================== //
  const int64_t max_total_work = hkv * tq * topk;
  const int64_t nqps = std::max<int64_t>(1, (max_total_work + grid_size - 1) / grid_size);
  auto work_start = at::empty({grid_size, 3}, i32);
  auto work_end = at::empty({grid_size, 3}, i32);
  {
    // Fed the COMPACT CSR (sel_offsets, plateaued at the head total): the scheduler binary-
    // searches it and emits COMPACT-j block indices (its head_base logic is unchanged).
    ArgList a;
    a.add_tensor(sel_offsets);
    a.add_tensor(work_start);
    a.add_tensor(work_end);
    a.add_i32(static_cast<int32_t>(num_block_slots));  // nbs
    a.add_i64(nqps);
    a.add_stream(stream);
    a.run(H.fn("scheduler"));
  }

  // =============================== forward =============================== //
  auto o_flat = at::empty({hkv * tq * topk * qhead, d}, at::TensorOptions().dtype(H.partial_dtype).device(device));
  auto m_partial = at::empty({hkv, seg}, f32);
  auto l_partial = at::empty({hkv, seg}, f32);
  auto k_perm = k_cache.permute({0, 2, 1, 3});
  auto v_perm = v_cache.permute({0, 2, 1, 3});
  {
    ArgList a;
    a.add_tensor(q);
    a.add_tensor(k_perm);
    a.add_tensor(v_perm);
    a.add_tensor(o_flat);  // mO (only element_type matters)
    a.add_tensor(m_partial);
    a.add_tensor(l_partial);
    a.add_tensor(slot);
    a.add_tensor(sel_offsets);  // COMPACT CSR (mKvToQOffsets)
    a.add_tensor(idx_ranks);
    a.add_tensor(work_start);
    a.add_tensor(work_end);
    a.add_tensor(sel_slots);  // mSelSlots
    a.add_tensor(num_sel);    // mNumSel
    a.add_i32(static_cast<int32_t>(grid_size));
    a.add_tensor(cuq);
    a.add_tensor(sk);
    a.add_i32(static_cast<int32_t>(n_batches));
    a.add_f32(static_cast<float>(softmax_scale));
    a.add_tensor(o_flat);  // mO2d
    a.add_stream(stream);
    a.run(H.fn("forward"));
  }

  // =============================== combine =============================== //
  auto lp = m_partial.reshape({-1});
  auto ll = l_partial.reshape({-1});
  at::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value();
    TORCH_CHECK(out.dim() == 3 && out.size(0) == tq && out.size(1) == hq && out.size(2) == d,
                "out must be [Tq, Hq, D]");
    TORCH_CHECK(out.scalar_type() == H.out_dtype, "out dtype must match the handle's out_dtype");
    TORCH_CHECK(out.is_contiguous() && out.device() == device, "out must be contiguous on q's device");
  } else {
    out = at::empty({tq, hq, d}, at::TensorOptions().dtype(H.out_dtype).device(device));
  }
  auto out_b = out.unsqueeze(0);
  at::Tensor lse;
  {
    ArgList a;
    a.add_tensor(o_flat);    // mO_partial
    a.add_tensor(lp);        // mLSE_partial (m~)
    a.add_tensor(out_b);     // mO
    a.add_tensor(ll);        // mL_partial (l)
    a.add_tensor(inv);       // mInv
    if (H.return_lse) {
      lse = at::empty({1, hq, tq}, f32);
      a.add_tensor(lse);     // mLSE
    }
    a.add_stream(stream);
    a.run(H.fn("combine"));
  }

  at::Tensor lse_out = H.return_lse ? lse.squeeze(0) : at::Tensor();
  return std::make_tuple(out, lse_out);
}

}  // namespace fmha_sm100
