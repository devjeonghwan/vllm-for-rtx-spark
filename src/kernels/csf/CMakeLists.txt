cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0141 NEW)
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
project(mbx_original_csf_native_host LANGUAGES CXX)
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Native Windows ARM64 required")
endif()
if(NOT EXISTS "${TORCH_ROOT}/include/torch/csrc/stable/library.h")
  message(FATAL_ERROR "Explicit task Torch root with stable ABI headers required")
endif()
if(NOT EXISTS "${CUDA_ROOT}/include/cuda.h")
  message(FATAL_ERROR "Explicit CUDA SDK required; no CUDA compilation performed")
endif()
add_library(mbx_original_csf_native_host SHARED native_host.cpp)
target_compile_features(mbx_original_csf_native_host PRIVATE cxx_std_17)
target_compile_options(mbx_original_csf_native_host PRIVATE /EHsc /W3)
target_link_options(mbx_original_csf_native_host PRIVATE
  "/MAP:${CMAKE_CURRENT_BINARY_DIR}/native-host.map")
target_compile_definitions(mbx_original_csf_native_host PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN
  TORCH_TARGET_VERSION=0x020D000000000000ULL USE_CUDA)
target_include_directories(mbx_original_csf_native_host PRIVATE
  "${TORCH_ROOT}/include" "${CUDA_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(mbx_original_csf_native_host PRIVATE
  "${TORCH_ROOT}/lib/torch_cpu.lib" "${TORCH_ROOT}/lib/torch_cuda.lib"
  "${TORCH_ROOT}/lib/c10.lib" "${TORCH_ROOT}/lib/c10_cuda.lib"
  "${CUDA_ARM64_LIB}/cuda.lib" "${CUDA_ARM64_LIB}/cudart.lib")
set_target_properties(mbx_original_csf_native_host PROPERTIES
  MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
# No Python/pybind, no find_package(Torch), no Python execution, no NVCC/device sources.
