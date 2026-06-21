// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/ops/RoPEOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLRoPEOp::OpenCLRoPEOp(const aops::RoPEOpOptions& options) : aops::RoPEOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  kernel_f32_ = runtime->buildKernel("rope", "rope_f32", {});
  MLLM_RT_ASSERT(kernel_f32_);

  kernel_f16_ = runtime->buildKernel("rope", "rope_f16", {});
  MLLM_RT_ASSERT(kernel_f16_);
}

void OpenCLRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& sin_table = inputs[1];
  auto& cos_table = inputs[2];
  auto& output = outputs[0];

  int B = input.shape()[0];
  int H = input.shape()[1];
  int S = input.shape()[2];
  int D = input.shape()[3];

  cl::NDRange global_work_size = {static_cast<size_t>(D / 2), static_cast<size_t>(S), static_cast<size_t>(B * H)};

  std::shared_ptr<KernelWrap> kernel;
  if (input.dtype() == kFloat32) {
    kernel = kernel_f32_;
  } else if (input.dtype() == kFloat16) {
    kernel = kernel_f16_;
  } else {
    MLLM_ERROR("OpenCLRoPE doesn't support type {}", input.dtype());
    return;
  }

  cl_mem src_mem = (cl_mem)input.impl()->storage()->ptr_;
  cl_mem dst_mem = (cl_mem)output.impl()->storage()->ptr_;
  cl_mem sin_mem = (cl_mem)sin_table.impl()->storage()->ptr_;
  cl_mem cos_mem = (cl_mem)cos_table.impl()->storage()->ptr_;

  cl_int ret = CL_SUCCESS;
  ret |= kernel->get().setArg(0, sizeof(cl_mem), &src_mem);
  ret |= kernel->get().setArg(1, sizeof(cl_mem), &dst_mem);
  ret |= kernel->get().setArg(2, sizeof(cl_mem), &sin_mem);
  ret |= kernel->get().setArg(3, sizeof(cl_mem), &cos_mem);
  ret |= kernel->get().setArg(4, sizeof(int), &H);
  ret |= kernel->get().setArg(5, sizeof(int), &S);
  ret |= kernel->get().setArg(6, sizeof(int), &D);

  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLRoPEOp setArg failed: {}", ret); }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange, global_work_size);
  if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute rope kernel, error code: {}", error); }
}

}  // namespace mllm::opencl