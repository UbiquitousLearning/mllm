// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/OpenCLDispatcher.hpp"
#include "mllm/backends/opencl/ops/AddOp.hpp"
#include "mllm/backends/opencl/ops/EmbeddingOp.hpp"
#include "mllm/backends/opencl/ops/FillOp.hpp"
#include "mllm/backends/opencl/ops/GraphOps.hpp"
#include "mllm/backends/opencl/ops/RMSNormOp.hpp"
#include "mllm/backends/opencl/ops/RoPEOp.hpp"
#include "mllm/backends/opencl/ops/SliceOp.hpp"
#include "mllm/backends/opencl/ops/TransposeOp.hpp"
#include "mllm/backends/opencl/ops/ViewOp.hpp"
#include "mllm/backends/opencl/ops/X2XOp.hpp"

#include "mllm/mllm.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/backends/opencl/OpenCLAllocator.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include <memory>
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm {

void initOpenCLBackend() {
  auto& ctx = Context::instance();

  // 1. Register backend
  auto opencl_backend = std::make_shared<opencl::OpenCLBackend>();
  ctx.registerBackend(opencl_backend);

  // 2. Initialize memory manager
  ctx.memoryManager()->registerAllocator(kOpenCL, opencl_backend->allocator(), MemoryManagerOptions());

  ctx.dispatcherManager()->registerDispatcher(
      createOpenCLDispatcher(ctx.dispatcherManager()->getExecutor(), opencl::OpenCLDispatcherOptions()));
}

namespace opencl {

OpenCLBackend::OpenCLBackend() : Backend(kOpenCL, nullptr) {
  regOpFactory<OpenCLX2XOpFactory, OpenCLAddOpFactory, OpenCLEmbeddingOpFactory, OpenCLGraphBeginOpFactory,
               OpenCLGraphEndOpFactory, OpenCLSliceOpFactory, OpenCLViewOpFactory, OpenCLTransposeOpFactory, OpenCLX2XOpFactory,
               OpenCLFillOpFactory, OpenCLRMSNormOpFactory, OpenCLRoPEOpFactory>();

  runtime_ = std::shared_ptr<OpenCLRuntime>(new OpenCLRuntime());
  if (!runtime_) {
    MLLM_ERROR("Failed to initialize OpenCL runtime.\n");
    return;
  }
  if (runtime_->getDevices().empty()) {
    MLLM_ERROR("No OpenCL devices found.\n");
    return;
  }

  this->allocator_ = std::make_shared<OpenCLAllocator>(runtime_);

  const auto& device = runtime_->getDevices()[0];
  std::string device_name;
  device.getInfo(CL_DEVICE_NAME, &device_name);

  MLLM_INFO("Initializing OpenCL backend..");
  MLLM_INFO("Device: {}", device_name.c_str());
  MLLM_INFO("GPUType: {}", (int)runtime_->getGpuType());
  MLLM_INFO("Global Memory Cache Size: {} KB", runtime_->deviceGlobalMemeryCacheSize() / 1024);
  MLLM_INFO("Compute Units: {}", runtime_->deviceComputeUnits());
  MLLM_INFO("Max Work Group Size: {}", runtime_->maxWorkGroupSize());
  MLLM_INFO("FP16 Supported: {}", runtime_->isSupportedFP16() ? "Yes" : "No");
  MLLM_INFO("Int8 dot Supported: {}", runtime_->isSupportedDotInt8() ? "Yes" : "No");
  MLLM_INFO("Int8 dot Accumulate Supported: {}", runtime_->isSupportedDotAccInt8() ? "Yes" : "No");
}

}  // namespace opencl
}  // namespace mllm
