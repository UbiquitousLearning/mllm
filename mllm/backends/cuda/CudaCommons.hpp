// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include <nvml.h>
#include <cuda_runtime.h>
#include "mllm/utils/Common.hpp"

#define MLLM_CUDA_CHECK(err)                                                                                    \
  if (err != cudaSuccess) {                                                                                     \
    MLLM_ERROR_EXIT(::mllm::ExitCode::kCudaError, "CUDA error code {}: {}", int(err), cudaGetErrorString(err)); \
  }

#define MLLM_NVML_CHECK(err) \
  if (err != NVML_SUCCESS) { MLLM_ERROR_EXIT(::mllm::ExitCode::kCudaError, "NVML error: {}", nvmlErrorString(err)); }

namespace mllm::cuda {

struct NvGpuInfo {
  std::string name;
  unsigned int id;
  unsigned int sm_count;
  unsigned int cuda_core_per_sm;
  unsigned int tensor_core_per_sm;
  unsigned int l1_cache;           // bytes
  unsigned int shared_mem_per_sm;  // bytes
  unsigned int global_mem;         // bytes
  unsigned int max_thread_per_sm;
  unsigned int warp_size_in_thread;
  unsigned int architecture;
};

class NvGpuMetaInfo {
 public:
  NvGpuMetaInfo();

  static NvGpuMetaInfo& instance() {
    static NvGpuMetaInfo instance;
    return instance;
  }

  NvGpuMetaInfo(const NvGpuMetaInfo&) = delete;
  NvGpuMetaInfo& operator=(const NvGpuMetaInfo&) = delete;

  std::vector<NvGpuInfo> devices;
};

}  // namespace mllm::cuda
