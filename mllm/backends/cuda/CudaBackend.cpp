// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cuda/CudaAllocator.hpp"
#include "mllm/backends/cuda/CudaCommons.hpp"
#include "mllm/backends/cuda/CudaBackend.hpp"

namespace mllm::cuda {

CudaBackend::CudaBackend() : Backend(kCUDA, createCudaAllocator()) {
  NvGpuMetaInfo::instance();
  auto& devices = NvGpuMetaInfo::instance().devices;
  for (auto& d : devices) { MLLM_INFO("Found device: {}", d.name); }

  // regOpFactory<>();
}

std::shared_ptr<CudaBackend> createCudaBackend() { return std::make_shared<CudaBackend>(); }

}  // namespace mllm::cuda
