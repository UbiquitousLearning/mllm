// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "CausalMaskOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLCausalMaskOp::OpenCLCausalMaskOp(const aops::CausalMaskOpOptions& options) : aops::CausalMaskOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  kernel_fp32_ = runtime->buildKernel("causal_mask", "causal_mask_fp32", {});
  MLLM_RT_ASSERT(kernel_fp32_);
  kernel_fp16_ = runtime->buildKernel("causal_mask", "causal_mask_fp16", {});
  MLLM_RT_ASSERT(kernel_fp16_);
}

void OpenCLCausalMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto shape = input.shape();
  int B = shape[0];
  int H = shape[1];
  int S = shape[2];
  int D = shape[3];

  int sliding_window = options_.sliding_window ? 1 : 0;
  int window_size = options_.window_size;

  auto cl_buffer_in = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_out = (cl_mem)output.impl()->storage()->ptr_;

  cl_int ret = CL_SUCCESS;
  std::shared_ptr<KernelWrap> kernel = nullptr;

  if (input.dtype() == mllm::kFloat32) {
    kernel = kernel_fp32_;
  } else if (input.dtype() == mllm::kFloat16) {
    kernel = kernel_fp16_;
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLCausalMaskOp supports only FP32 and FP16");
  }

  ret |= kernel->get().setArg(0, sizeof(cl_mem), &cl_buffer_in);
  ret |= kernel->get().setArg(1, sizeof(cl_mem), &cl_buffer_out);
  ret |= kernel->get().setArg(2, sizeof(int), &B);
  ret |= kernel->get().setArg(3, sizeof(int), &H);
  ret |= kernel->get().setArg(4, sizeof(int), &S);
  ret |= kernel->get().setArg(5, sizeof(int), &D);
  ret |= kernel->get().setArg(6, sizeof(int), &sliding_window);
  ret |= kernel->get().setArg(7, sizeof(int), &window_size);

  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLCausalMaskOp setArg failed: {}", ret); }

  // Global size: [D, S, B*H]
  cl::NDRange global_size(D, S, B * H);

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange, global_size, cl::NullRange);
  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLCausalMaskOp enqueueNDRangeKernel failed: {}", error);
  }
}

}  // namespace mllm::opencl
