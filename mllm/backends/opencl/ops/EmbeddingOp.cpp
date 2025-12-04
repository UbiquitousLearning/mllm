// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/backends/opencl/ops/EmbeddingOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::opencl {

OpenCLEmbeddingOp::OpenCLEmbeddingOp(const aops::EmbeddingOpOptions& options) : aops::EmbeddingOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  kernel_fp32_buffer_ = runtime->buildKernel("embedding", "embedding_fp32", {});
  MLLM_RT_ASSERT(kernel_fp32_buffer_);
  kernel_q40_buffer_ = runtime->buildKernel("embedding", "embedding_q4_0", {});
  MLLM_RT_ASSERT(kernel_q40_buffer_);
}

void OpenCLEmbeddingOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  std::shared_ptr<KernelWrap> kernel_wrapper;

  switch (weight_.dtype()) {
    case DataTypes::kFloat32: kernel_wrapper = kernel_fp32_buffer_; break;
    case DataTypes::kGGUF_Q4_0: kernel_wrapper = kernel_q40_buffer_; break;
    default: MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Unsupported weight data type in EmbeddingOp: {}", weight_.dtype());
  }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_input = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_weight = (cl_mem)weight_.impl()->storage()->ptr_;
  auto cl_buffer_output = (cl_mem)output.impl()->storage()->ptr_;

  cl_int ret = CL_SUCCESS;
  ret |= kernel_wrapper->get().setArg(0, sizeof(cl_mem), &cl_buffer_input);
  ret |= kernel_wrapper->get().setArg(1, sizeof(cl_mem), &cl_buffer_weight);
  ret |= kernel_wrapper->get().setArg(2, sizeof(cl_mem), &cl_buffer_output);

  int sequence_len = input.shape()[1];

  ret |= kernel_wrapper->get().setArg(3, sizeof(int), &options_.vocab_size);
  ret |= kernel_wrapper->get().setArg(4, sizeof(int), &options_.hidden_size);
  ret |= kernel_wrapper->get().setArg(5, sizeof(int), &sequence_len);

  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLEmbeddingOp setArg failed: {}", ret); }

  size_t global_size[2] = {static_cast<size_t>(options_.hidden_size), static_cast<size_t>(sequence_len)};

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_wrapper->get(), cl::NullRange,
                                                            cl::NDRange(global_size[0], global_size[1]), cl::NullRange);

  if (error != CL_SUCCESS) { MLLM_ERROR("Failed to execute embedding kernel, error code: {}", error); }

  runtime->commandQueue().finish();
}
}  // namespace mllm::opencl