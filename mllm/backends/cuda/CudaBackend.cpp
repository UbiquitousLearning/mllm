// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/Context.hpp"
#include "mllm/backends/cuda/CudaAllocator.hpp"
#include "mllm/backends/cuda/CudaCommons.hpp"
#include "mllm/backends/cuda/CudaBackend.hpp"

namespace mllm {

void initCudaBackend() {
  auto& ctx = Context::instance();

  // 1. Register host backend
  auto cuda_backend = cuda::createCudaBackend();
  ctx.registerBackend(cuda_backend);

  // 2. Initialize memory manager
  ctx.memoryManager()->registerAllocator(kCUDA, cuda_backend->allocator(), MemoryManagerOptions());
}

namespace cuda {

CudaBackend::CudaBackend() : Backend(kCUDA, createCudaAllocator()) {
  NvGpuMetaInfo::instance();
  auto& devices = NvGpuMetaInfo::instance().devices;
  for (auto& d : devices) { MLLM_INFO("Found device: {}", d.name); }

  // regOpFactory<>();
}

std::shared_ptr<CudaBackend> createCudaBackend() { return std::make_shared<CudaBackend>(); }

}  // namespace cuda
}  // namespace mllm
