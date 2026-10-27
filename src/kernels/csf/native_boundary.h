// Windows-only CUDA host boundary. Original CUDA device bytes remain unchanged.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include "original/core/scalar_type.hpp"
#include "original/torch_utils.h"

static_assert(sizeof(void*) == 8 && sizeof(int) == 4 && sizeof(bool) == 1);
static_assert(sizeof(int64_t) == 8);
static_assert(vllm::kBFloat16.id() == 1125899906909960LL);
static_assert(vllm::kFE2M1f.id() == 562949953487106LL);
static_assert(vllm::kFE4M3fn.id() == 2814749767172868LL);
// NVCC exposes global min/max; MSVC host compilation needs explicit aliases.
using std::min;
using std::max;

namespace mbx_marlin_csf_native {
// Exact public marlin.cuh host constants; device headers are not compiled.
static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;
static constexpr int tile_size = 16;
constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

using MarlinFuncPtr = CUfunction;
inline constexpr MarlinFuncPtr MarlinDefault = nullptr;

inline void driver_check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  STD_TORCH_CHECK(false, operation, " failed: ", name ? name : "unknown CUDA error");
}

struct ModuleEntry {
  CUmodule module = nullptr;
  std::wstring cubin_path;
  std::vector<uint8_t> cubin_bytes;
  std::map<std::string, CUfunction> functions;
};

#include "kernel_symbols.inc"

inline std::wstring cubin_path() {
  DWORD length = GetEnvironmentVariableW(L"MBX_MARLIN_CSF_CUBIN", nullptr, 0);
  STD_TORCH_CHECK(length > 1, "MBX_MARLIN_CSF_CUBIN must name the pinned original cubin");
  std::vector<wchar_t> path(length);
  STD_TORCH_CHECK(GetEnvironmentVariableW(L"MBX_MARLIN_CSF_CUBIN", path.data(), length)
                      == length - 1, "CUBIN path changed while reading it");
  return path.data();
}

inline std::vector<uint8_t> read_cubin(const std::wstring& path) {
  HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  STD_TORCH_CHECK(file != INVALID_HANDLE_VALUE, "Cannot open pinned original cubin");
  struct CloseFile { HANDLE file; ~CloseFile() { CloseHandle(file); } } close{file};
  LARGE_INTEGER size;
  STD_TORCH_CHECK(GetFileSizeEx(file, &size) && size.QuadPart == 1748200,
                  "Pinned original cubin byte count changed");
  std::vector<uint8_t> bytes(static_cast<size_t>(size.QuadPart));
  DWORD read = 0;
  STD_TORCH_CHECK(ReadFile(file, bytes.data(), static_cast<DWORD>(bytes.size()),
                           &read, nullptr) && read == bytes.size(),
                  "Cannot read complete original cubin");
  STD_TORCH_CHECK(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' &&
                      bytes[3] == 'F' && bytes[18] == 190 && bytes[19] == 0,
                  "Expected CUDA DEVICE ELF, never a Linux HOST library");
  return bytes;
}

inline CUfunction original_function(const char* name) {
  CUcontext context = nullptr;
  driver_check(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  STD_TORCH_CHECK(context != nullptr, "Torch CUDA context must already be current");
  // Deliberately process-pinned, like the original NVCC fatbinary registration.
  // No Python Graph lease, scratch cache or module unload callback is introduced.
  static auto* entries = new std::map<CUcontext, ModuleEntry>();
  static auto* mutex = new std::mutex();
  std::lock_guard<std::mutex> lock(*mutex);
  auto existing = entries->find(context);
  if (existing == entries->end()) {
    void* current = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(-1, &current));
    CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
    driver_check(cuStreamIsCapturing(reinterpret_cast<CUstream>(current), &capture),
                 "cuStreamIsCapturing");
    STD_TORCH_CHECK(capture == CU_STREAM_CAPTURE_STATUS_NONE,
                    "Original cubin must be eagerly warmed before Graph capture");
    HMODULE self = nullptr;
    static const char anchor = 0;
    STD_TORCH_CHECK(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                          GET_MODULE_HANDLE_EX_FLAG_PIN,
                                      reinterpret_cast<LPCWSTR>(&anchor), &self),
                    "Cannot pin native host DLL");
    ModuleEntry entry;
    entry.cubin_path = cubin_path();
    entry.cubin_bytes = read_cubin(entry.cubin_path);
    driver_check(cuModuleLoadData(&entry.module, entry.cubin_bytes.data()),
                 "cuModuleLoadData original cubin");
    // All original functions are resolved before capture, not lazily per shape.
    // Preserve a loaded module if resolution fails; never unload a potentially
    // referenced CUfunction. The failed slot remains process-owned and rejected.
    existing = entries->emplace(context, std::move(entry)).first;
    for (const char* symbol : original_kernel_symbols) {
      CUfunction handle = nullptr;
      driver_check(cuModuleGetFunction(&handle, existing->second.module, symbol),
                   "cuModuleGetFunction original 15 symbols");
      existing->second.functions.emplace(symbol, handle);
    }
  }
  auto& entry = existing->second;
  STD_TORCH_CHECK(entry.functions.size() == 15, "Original module initialization incomplete");
  STD_TORCH_CHECK(entry.cubin_path == cubin_path(), "Pinned cubin path changed in this context");
  auto function = entry.functions.find(name);
  STD_TORCH_CHECK(function != entry.functions.end(), "Kernel absent from pinned original 15 symbols");
  return function->second;
}

// Identical source selector reads attr.numRegs; only the API boundary changes.
inline CUresult cudaFuncGetAttributes(cudaFuncAttributes* attr, CUfunction function) {
  int registers = 0;
  driver_check(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function),
               "cuFuncGetAttribute NUM_REGS");
  *attr = {};
  attr->numRegs = registers;
  return CUDA_SUCCESS;
}

inline CUresult cudaFuncSetAttribute(CUfunction function, cudaFuncAttribute attribute,
                                     int value) {
  STD_TORCH_CHECK(attribute == cudaFuncAttributeMaxDynamicSharedMemorySize,
                  "Unexpected original function attribute");
  driver_check(cuFuncSetAttribute(function,
                                 CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value),
               "cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES");
  return CUDA_SUCCESS;
}

inline void driver_launch(CUfunction function, int blocks, int threads, int shared,
                          cudaStream_t stream, void** arguments) {
  driver_check(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, shared,
                              reinterpret_cast<CUstream>(stream), arguments, nullptr),
               "cuLaunchKernel original CSF 25-argument kernel");
}
}  // namespace mbx_marlin_csf_native
