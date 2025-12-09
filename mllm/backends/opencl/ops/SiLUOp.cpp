// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "SiLUOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

OpenCLSiLUOp::OpenCLSiLUOp(const aops::SiLUOpOptions& options) : aops::SiLUOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("silu", "silu_fp32", {});
  kernel_fp16_buffer_ = runtime->buildKernel("silu", "silu_fp16", {});
}

void OpenCLSiLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_in = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_out = (cl_mem)output.impl()->storage()->ptr_;

  size_t global_size = input.numel();

  if (input.dtype() == MLLM_TYPE_F32) {
    cl_int ret = CL_SUCCESS;
    ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_in);
    ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_out);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLSiLUOp setArg failed: {}", ret); }

    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute silu_fp32 kernel, error code: {}", error);
    }
  } else if (input.dtype() == MLLM_TYPE_F16) {
    size_t global_size_fp16 = (global_size + 3) / 4;
    int count = global_size;

    cl_int ret = CL_SUCCESS;
    ret |= kernel_fp16_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_in);
    ret |= kernel_fp16_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_out);
    ret |= kernel_fp16_buffer_->get().setArg(2, sizeof(int), &count);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLSiLUOp setArg failed: {}", ret); }

    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp16_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size_fp16), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute silu_fp16 kernel, error code: {}", error);
    }
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLSiLUOp not support dtype: {}", input.dtype());
  }
}

}  // namespace mllm::opencl
