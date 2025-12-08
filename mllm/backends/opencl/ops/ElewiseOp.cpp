// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "ElewiseOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

OpenCLAddOp::OpenCLAddOp(const aops::AddOpOptions& options) : aops::AddOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("add", "add_float", {});
}

void OpenCLAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input_a = inputs[0];
  auto& input_b = inputs[1];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_a = (cl_mem)input_a.impl()->storage()->ptr_;
  auto cl_buffer_b = (cl_mem)input_b.impl()->storage()->ptr_;
  auto cl_buffer_c = (cl_mem)output.impl()->storage()->ptr_;

  cl_int ret = CL_SUCCESS;
  ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
  ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
  ret |= kernel_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLAddOp setArg failed: {}", ret); }

  size_t global_size = input_a.numel();

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange, cl::NDRange(global_size),
                                                            cl::NullRange);

  if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute add kernel, error code: {}", error); }
}

OpenCLSubOp::OpenCLSubOp(const aops::SubOpOptions& options) : aops::SubOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("sub", "sub_float", {});
}

void OpenCLSubOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input_a = inputs[0];
  auto& input_b = inputs[1];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_a = (cl_mem)input_a.impl()->storage()->ptr_;
  auto cl_buffer_b = (cl_mem)input_b.impl()->storage()->ptr_;
  auto cl_buffer_c = (cl_mem)output.impl()->storage()->ptr_;

  cl_int ret = CL_SUCCESS;
  ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
  ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
  ret |= kernel_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLSubOp setArg failed: {}", ret); }

  size_t global_size = input_a.numel();

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange, cl::NDRange(global_size),
                                                            cl::NullRange);

  if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute sub kernel, error code: {}", error); }
}

OpenCLMulOp::OpenCLMulOp(const aops::MulOpOptions& options) : aops::MulOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("mul", "mul_float", {});
}

void OpenCLMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input_a = inputs[0];
  auto& input_b = inputs[1];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_a = (cl_mem)input_a.impl()->storage()->ptr_;
  auto cl_buffer_b = (cl_mem)input_b.impl()->storage()->ptr_;
  auto cl_buffer_c = (cl_mem)output.impl()->storage()->ptr_;

  cl_int ret = CL_SUCCESS;
  ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
  ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
  ret |= kernel_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLMulOp setArg failed: {}", ret); }

  size_t global_size = input_a.numel();

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange, cl::NDRange(global_size),
                                                            cl::NullRange);

  if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute mul kernel, error code: {}", error); }
}

OpenCLDivOp::OpenCLDivOp(const aops::DivOpOptions& options) : aops::DivOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("div", "div_float", {});
}

void OpenCLDivOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input_a = inputs[0];
  auto& input_b = inputs[1];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_a = (cl_mem)input_a.impl()->storage()->ptr_;
  auto cl_buffer_b = (cl_mem)input_b.impl()->storage()->ptr_;
  auto cl_buffer_c = (cl_mem)output.impl()->storage()->ptr_;

  cl_int ret = CL_SUCCESS;
  ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
  ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
  ret |= kernel_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLDivOp setArg failed: {}", ret); }

  size_t global_size = input_a.numel();

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange, cl::NDRange(global_size),
                                                            cl::NullRange);

  if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute div kernel, error code: {}", error); }
}

}  // namespace mllm::opencl