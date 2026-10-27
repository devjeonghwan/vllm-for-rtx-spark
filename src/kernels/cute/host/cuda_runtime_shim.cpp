// Native Windows ARM64 equivalents of the ORIGINAL CuTe4.8.1 CUDA wrappers.
// Original host COFF retains all TMA, parameter, tile, PDL and launch logic.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstddef>
#include <cstdint>

#if !defined(_M_ARM64)
#error Final CuTe host runtime must be built as native Windows ARM64.
#endif

static_assert(sizeof(void*) == 8);
static_assert(sizeof(cudaLaunchAttribute) == 72);
static_assert(sizeof(cudaLaunchConfig_t) == 56);
static_assert(offsetof(cudaLaunchConfig_t, stream) == 32);
static_assert(offsetof(cudaLaunchConfig_t, attrs) == 40);
static_assert(offsetof(cudaLaunchConfig_t, numAttrs) == 48);

// Seven original wrappers are single tail jumps to these exact cudart APIs.
extern "C" cudaError_t _cudaLibraryLoadData(
    cudaLibrary_t* library, const void* code, cudaJitOption* jit_options,
    void** jit_values, unsigned int jit_count, cudaLibraryOption* library_options,
    void** library_values, unsigned int library_count) {
  return cudaLibraryLoadData(library, code, jit_options, jit_values, jit_count,
                             library_options, library_values, library_count);
}
extern "C" cudaError_t _cudaGetDevice(int* device) {
  return cudaGetDevice(device);
}
extern "C" cudaError_t _cudaDeviceGetAttribute(int* value, int attribute, int device) {
  return cudaDeviceGetAttribute(value, static_cast<cudaDeviceAttr>(attribute), device);
}
extern "C" cudaError_t _cudaLibraryGetKernel(cudaKernel_t* kernel, cudaLibrary_t library, const char* name) {
  return cudaLibraryGetKernel(kernel, library, name);
}
extern "C" cudaError_t _cudaFuncSetAttribute(const void* function, int attribute, int value) {
  return cudaFuncSetAttribute(function, static_cast<cudaFuncAttribute>(attribute), value);
}
extern "C" cudaError_t _cudaKernelSetAttributeForDevice(cudaKernel_t kernel, int attribute, int value, int device) {
  return cudaKernelSetAttributeForDevice(kernel, static_cast<cudaFuncAttribute>(attribute), value, device);
}
extern "C" cudaError_t _cudaLaunchKernelEx(const cudaLaunchConfig_t* config, const void* function, void** arguments) {
  return cudaLaunchKernelExC(config, function, arguments);
}

namespace {
std::atomic<void*> kernel_get_attribute{nullptr};
std::atomic<void*> device_get{nullptr};

template <typename Function>
Function original_driver_entry(std::atomic<void*>& cache, const char* name) {
  void* address = cache.load(std::memory_order_acquire);
  if (address == nullptr) {
    cudaDriverEntryPointQueryResult query_status{};
    // Exact original disassembly constant0x2f30=12080, flags0.
    const cudaError_t status = cudaGetDriverEntryPointByVersion(
        name, &address, 12080, 0, &query_status);
    if (status != cudaSuccess || static_cast<int>(query_status) != 0) {
      return nullptr;
    }
    void* expected = nullptr;
    cache.compare_exchange_strong(expected, address, std::memory_order_acq_rel);
    address = cache.load(std::memory_order_acquire);
  }
  return reinterpret_cast<Function>(address);
}
}  // namespace

// Original _cuKernelGetAttribute also converts the runtime device ordinal via
// cuDeviceGet before calling the Driver kernel API. It never gets CUfunction.
extern "C" int _cuKernelGetAttribute(int* value, int attribute, void* kernel, int ordinal) {
  using KernelAttribute = CUresult(CUDAAPI*)(int*, CUfunction_attribute, CUkernel, CUdevice);
  using DeviceGet = CUresult(CUDAAPI*)(CUdevice*, int);
  const auto get_attribute = original_driver_entry<KernelAttribute>(kernel_get_attribute, "cuKernelGetAttribute");
  if (get_attribute == nullptr) {
    return 999;
  }
  const auto get_device = original_driver_entry<DeviceGet>(device_get, "cuDeviceGet");
  if (get_device == nullptr) {
    return 999;
  }
  CUdevice device{};
  if (get_device(&device, ordinal) != CUDA_SUCCESS) {
    return 999;
  }
  return static_cast<int>(get_attribute(value, static_cast<CUfunction_attribute>(attribute),
                                       reinterpret_cast<CUkernel>(kernel), device));
}

// Minimal loader/cleanup helpers, not replacement host launch wrappers.
extern "C" __declspec(dllexport) int original_cute_runtime_abi(std::uint64_t* output, std::size_t count) {
  if (output == nullptr || count < 6) return -1;
  output[0] = 1;
  output[1] = sizeof(void*);
  output[2] = sizeof(cudaLaunchConfig_t);
  output[3] = sizeof(cudaLaunchAttribute);
  output[4] = CUDART_VERSION;
  output[5] = 0xaa64;
  return 0;
}
extern "C" __declspec(dllexport) cudaError_t original_cute_runtime_unload(cudaLibrary_t library) {
  return cudaLibraryUnload(library);
}
extern "C" __declspec(dllexport) cudaError_t original_cute_runtime_get_device(int* device) {
  return cudaGetDevice(device);
}
extern "C" __declspec(dllexport) const char* original_cute_runtime_error_string(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}
