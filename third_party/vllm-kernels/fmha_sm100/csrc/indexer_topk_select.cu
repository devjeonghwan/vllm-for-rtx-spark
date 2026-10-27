// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// Forced-tail TopK used by the Q8KV4/Q8KV8 paged indexers.

#include <limits>

#include "indexer_topk/m3_topk.cuh"
#include "tvm_ffi_utils.h"

namespace {

constexpr int64_t kMaximumColumns = m3::kMaxNvm;

template <typename T> T *tensor_data(TensorView tensor) {
  return reinterpret_cast<T *>(static_cast<char *>(tensor.data_ptr()) + tensor.byte_offset());
}

} // namespace

// ``lengths[row]`` counts the row's candidate pages including the forced local
// page, which is written to the final slot without being ranked. ``scores`` is
// ``[rows, cols]`` or a ``[groups, rows_per_group, cols]`` view whose rows are
// ranked in order, so a caller can select a strided subset of a score matrix.
// ``use_pdl`` launches with programmatic stream serialization; the Python
// wrapper defaults it to true and the prefill indexer turns it off.
void indexer_topk_select(TensorView scores, TensorView lengths, TensorView output,
                         bool use_pdl, int64_t stream_ptr) {
  CHECK_CUDA(scores);
  TVM_FFI_ICHECK(scores.ndim() == 2 || scores.ndim() == 3)
      << "scores must be [rows, cols] or [groups, rows_per_group, cols]";
  TVM_FFI_ICHECK(encode_dlpack_dtype(scores.dtype()) == float32_code) << "scores must be float32";
  int const col_dim = scores.ndim() - 1;
  int64_t const cols = scores.size(col_dim);
  int64_t const rows_per_group = scores.size(col_dim - 1);
  int64_t const row_stride = scores.stride(col_dim - 1);
  int64_t const groups = scores.ndim() == 3 ? scores.size(0) : 1;
  int64_t const group_stride = scores.ndim() == 3 ? scores.stride(0) : 0;
  TVM_FFI_ICHECK(groups > 0 && rows_per_group > 0 &&
                 groups * rows_per_group <= std::numeric_limits<int>::max())
      << "scores rows must be in [1, INT32_MAX]";
  TVM_FFI_ICHECK(cols > 0 && cols <= kMaximumColumns)
      << "scores columns must be in [1, " << kMaximumColumns << "]";
  TVM_FFI_ICHECK(scores.stride(col_dim) == 1) << "scores columns must have stride 1";
  TVM_FFI_ICHECK(row_stride >= cols && row_stride <= std::numeric_limits<int>::max())
      << "scores rows must not overlap and the row stride must fit in int32";
  TVM_FFI_ICHECK(groups == 1 || group_stride >= rows_per_group * row_stride)
      << "scores groups must not overlap";
  int64_t const rows = groups * rows_per_group;

  CHECK_INPUT_AND_TYPE(lengths, dl_int32);
  CHECK_DIM(1, lengths);
  TVM_FFI_ICHECK(lengths.size(0) == rows) << "lengths must have shape [rows]";
  CHECK_DEVICE(lengths, scores);

  CHECK_INPUT_AND_TYPE(output, dl_int32);
  CHECK_DIM(2, output);
  TVM_FFI_ICHECK(output.size(0) == rows && output.size(1) == m3::kTopK)
      << "output must have shape [rows, " << m3::kTopK << "]";
  CHECK_DEVICE(output, scores);

  ffi::CUDADeviceGuard device_guard(scores.device().device_id);
  m3::m3_launch(tensor_data<float const>(scores), tensor_data<int const>(lengths),
                tensor_data<int>(output), static_cast<int>(cols), static_cast<int>(row_stride),
                static_cast<int>(rows), static_cast<int>(rows_per_group), group_stride,
                reinterpret_cast<cudaStream_t>(stream_ptr), use_pdl);
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "indexer_topk_select launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(indexer_topk_select, indexer_topk_select);
