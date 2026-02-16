// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <mllm_kernel/utils.hpp>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#ifndef USE_ROCM
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;

using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
using fp8x2_e5m2_t = __nv_fp8x2_e5m2;

using fp32x4_t = float4;
#endif

namespace device {

#define MLLM_DEVICE __forceinline__ __device__

inline constexpr auto kWarpThreads = 32u;
inline constexpr auto kFullMask = 0xffffffffu;

template<bool kUsePDL>
MLLM_DEVICE void PDLWaitPrimary() {
#ifndef USE_ROCM
  if constexpr (kUsePDL) { asm volatile("griddepcontrol.wait;" ::: "memory"); }
#endif
}

template<bool kUsePDL>
MLLM_DEVICE void PDLTriggerSecondary() {
#ifndef USE_ROCM
  if constexpr (kUsePDL) { asm volatile("griddepcontrol.launch_dependents;" :::); }
#endif
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

template<typename T = char, std::integral... U>
MLLM_DEVICE auto offset(void* ptr, U... offset) -> void* {
  return static_cast<T*>(ptr) + (... + offset);
}

template<typename T = char, std::integral... U>
MLLM_DEVICE auto offset(const void* ptr, U... offset) -> const void* {
  return static_cast<const T*>(ptr) + (... + offset);
}

}  // namespace pointer

}  // namespace device

namespace mllm_kernel::host {

inline void RuntimeDeviceCheck(::cudaError_t error, DebugInfo location = {}) {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::mllm_kernel::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

inline void RuntimeDeviceCheck(DebugInfo location = {}) { return RuntimeDeviceCheck(::cudaGetLastError(), location); }

struct LaunchKernel {
 public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device, std::size_t dynamic_shared_mem_bytes = 0,
                        DebugInfo location = {}) noexcept
      : config_(s_make_config(grid_dim, block_dim, resolve_device(device), dynamic_shared_mem_bytes)), m_location(location) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, cudaStream_t stream, std::size_t dynamic_shared_mem_bytes = 0,
                        DebugInfo location = {}) noexcept
      : config_(s_make_config(grid_dim, block_dim, stream, dynamic_shared_mem_bytes)), m_location(location) {}

  LaunchKernel(const LaunchKernel&) = delete;
  LaunchKernel& operator=(const LaunchKernel&) = delete;

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  auto enable_pdl(bool enabled = true) -> LaunchKernel& {
    if (enabled) {
      m_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attrs[0].val.programmaticStreamSerializationAllowed = true;
      config_.numAttrs = 1;
      config_.attrs = m_attrs;
    } else {
      config_.numAttrs = 0;
    }
    return *this;
  }

  template<typename T, typename... Args>
  auto operator()(T&& kernel, Args&&... args) const -> void {
    RuntimeDeviceCheck(::cudaLaunchKernelEx(&config_, kernel, std::forward<Args>(args)...), m_location);
  }

 private:
  static auto s_make_config(  // Make a config for kernel launch
      dim3 grid_dim, dim3 block_dim, cudaStream_t stream, std::size_t smem) -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }

  cudaLaunchConfig_t config_;
  const DebugInfo m_location;
  cudaLaunchAttribute m_attrs[1];
};

}  // namespace mllm_kernel::host
