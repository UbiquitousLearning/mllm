// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/ops/CopyOp.hpp"
#include <cstdint>

#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLCopyOp::OpenCLCopyOp(const aops::CopyOpOptions& options) : aops::CopyOp(options) {}

void OpenCLCopyOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& src = inputs[0];
  auto& dst = inputs[1];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_src = cl::Buffer((cl_mem)src.impl()->storage()->ptr_, true);
  auto cl_buffer_dst = cl::Buffer((cl_mem)dst.impl()->storage()->ptr_, true);

  int64_t src_offset = src.impl()->storageOffset() / lanesOfType(src.dtype()) * bytesOfType(src.dtype());
  int64_t dst_offset = dst.impl()->storageOffset() / lanesOfType(dst.dtype()) * bytesOfType(dst.dtype());
  int64_t total_bytes = src.numel() / lanesOfType(src.dtype()) * bytesOfType(src.dtype());

  cl::CommandQueue& queue = runtime->commandQueue();

  cl_int err = queue.enqueueCopyBuffer(cl_buffer_src, cl_buffer_dst,
                                       static_cast<size_t>(src_offset),  // src offset in bytes
                                       static_cast<size_t>(dst_offset),  // dst offset in bytes
                                       static_cast<size_t>(total_bytes));

  if (err != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "enqueueCopyBuffer failed: {}", err); }
}

}  // namespace mllm::opencl
