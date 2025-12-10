// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "SoftmaxOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

OpenCLSoftmaxOp::OpenCLSoftmaxOp(const aops::SoftmaxOpOptions& options) : aops::SoftmaxOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  kernel_fp32_ = runtime->buildKernel("softmax", "softmax_fp32", {});
  MLLM_RT_ASSERT(kernel_fp32_);
  kernel_fp16_ = runtime->buildKernel("softmax", "softmax_fp16", {});
  MLLM_RT_ASSERT(kernel_fp16_);
}

void OpenCLSoftmaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  int D = input.shape()[input.shape().size() - 1];
  int num_rows = input.numel() / D;

  auto cl_buffer_in = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_out = (cl_mem)output.impl()->storage()->ptr_;

  cl_int ret = CL_SUCCESS;
  std::shared_ptr<KernelWrap> kernel = nullptr;

  if (input.dtype() == mllm::kFloat32) {
    kernel = kernel_fp32_;
  } else if (input.dtype() == mllm::kFloat16) {
    kernel = kernel_fp16_;
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLSoftmaxOp supports only FP32 and FP16");
  }

  ret |= kernel->get().setArg(0, sizeof(cl_mem), &cl_buffer_in);
  ret |= kernel->get().setArg(1, sizeof(cl_mem), &cl_buffer_out);
  ret |= kernel->get().setArg(2, sizeof(int), &D);

  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLSoftmaxOp setArg failed: {}", ret); }

  size_t local_size = 256;
  size_t global_size = num_rows * local_size;

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange, cl::NDRange(global_size),
                                                            cl::NDRange(local_size));

  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute softmax kernel, error code: {}", error);
  }
}

}  // namespace mllm::opencl
