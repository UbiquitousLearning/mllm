// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "FillOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

OpenCLFillOp::OpenCLFillOp(const aops::FillOpOptions& options) : aops::FillOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("fill", "fill_fp32", {});
  kernel_arange_fp32_buffer_ = runtime->buildKernel("fill", "fill_arange_fp32", {});
}

void OpenCLFillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& output = outputs[0];
  if (output.dtype() != MLLM_TYPE_F32) {
    MLLM_ERROR("OpenCLFillOp only supports FP32 currently.");
    return;
  }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  auto cl_buffer_dst = (cl_mem)output.impl()->storage()->ptr_;
  size_t global_size = output.numel();

  cl_int ret = CL_SUCCESS;

  if (options_.type == aops::FillOpTypes::kZeros) {
    float value = 0.0f;
    ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(float), &value);
    ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_dst);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLFillOp setArg failed: {}", ret); }
    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute fill kernel, error code: {}", error);
    }
  } else if (options_.type == aops::FillOpTypes::kOnes) {
    float value = 1.0f;
    ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(float), &value);
    ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_dst);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLFillOp setArg failed: {}", ret); }
    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute fill kernel, error code: {}", error);
    }
  } else if (options_.type == aops::FillOpTypes::kSpecific) {
    float value = options_.value;
    ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(float), &value);
    ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_dst);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLFillOp setArg failed: {}", ret); }
    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute fill kernel, error code: {}", error);
    }
  } else if (options_.type == aops::FillOpTypes::kArange) {
    float start = options_.start;
    float step = options_.step;
    ret |= kernel_arange_fp32_buffer_->get().setArg(0, sizeof(float), &start);
    ret |= kernel_arange_fp32_buffer_->get().setArg(1, sizeof(float), &step);
    ret |= kernel_arange_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_dst);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLFillOp setArg failed: {}", ret); }
    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_arange_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute fill_arange kernel, error code: {}", error);
    }
  } else {
    MLLM_ERROR("OpenCLFillOp not implemented for type: {}", (int)options_.type);
  }
}
}  // namespace mllm::opencl
