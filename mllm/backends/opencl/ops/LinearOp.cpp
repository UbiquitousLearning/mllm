// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/ops/LinearOp.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/mllm.hpp"

namespace mllm::opencl {

OpenCLLinearOp::OpenCLLinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(Context::instance().getBackend(kOpenCL))->runtime();
  std::string program_name = "matmul_transb_bias";

  std::set<std::string> buildOptions;
  if (std::static_pointer_cast<OpenCLBackend>(Context::instance().getBackend(kOpenCL))->runtime()->isSupportedFP16()) {
    buildOptions.insert("-DSUPPORTS_FP16");
  }

  // Build kernels
  kernel_fp32_transb_bias_ = runtime->buildKernel(program_name, "gemm_fp32_transb_bias", buildOptions);
  MLLM_RT_ASSERT(kernel_fp32_transb_bias_);

  kernel_fp32_q4_0_transb_bias_ = runtime->buildKernel(program_name, "gemm_fp32_q4_0_transb_bias", buildOptions);
  MLLM_RT_ASSERT(kernel_fp32_q4_0_transb_bias_);

  kernel_gemv_fp32_q4_0_transb_bias_ = runtime->buildKernel(program_name, "gemv_fp32_q4_0_transb_bias", buildOptions);
  MLLM_RT_ASSERT(kernel_gemv_fp32_q4_0_transb_bias_);
}

void OpenCLLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto input_shape = input.shape();
  MLLM_RT_ASSERT(input_shape.size() >= 2);

  // In Linear
  // inputs is always: [..., S, in_channels]
  // outputs is always: [..., S, out_channels]
  int M = input_shape[input_shape.size() - 2];
  int K = input_shape[input_shape.size() - 1];
  int N = options_.out_channels;
  MLLM_RT_ASSERT_EQ(K, options_.in_channels);

  int batch_count = 1;
  for (size_t i = 0; i < input_shape.size() - 2; ++i) { batch_count *= input_shape[i]; }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_input = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_weight = (cl_mem)weight_.impl()->storage()->ptr_;
  auto cl_buffer_output = (cl_mem)output.impl()->storage()->ptr_;
  cl_mem cl_buffer_bias = cl_buffer_input;  // dummy init
  int has_bias = 0;
  if (options_.bias) {
    cl_buffer_bias = (cl_mem)bias_.impl()->storage()->ptr_;
    has_bias = 1;
  }

  cl_int ret = CL_SUCCESS;
  std::shared_ptr<KernelWrap> kernel_wrapper;
  cl::NDRange global_size;
  cl::NDRange local_size = cl::NullRange;
  cl_uint index = 0;

  if (weight_.dtype() == DataTypes::kFloat32) {
    kernel_wrapper = kernel_fp32_transb_bias_;

    ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_input);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_weight);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_bias);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_output);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &M);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &K);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &N);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &has_bias);

    // TILE_SIZE = 16
    int tile_size = 16;
    int gws_0 = (N + tile_size - 1) / tile_size * tile_size;
    int gws_1 = (M + tile_size - 1) / tile_size * tile_size;
    int gws_2 = batch_count;

    global_size = cl::NDRange(gws_0, gws_1, gws_2);
    local_size = cl::NDRange(tile_size, tile_size, 1);

  } else if (weight_.dtype() == DataTypes::kGGUF_Q4_0) {
    if (M == 1) {
      kernel_wrapper = kernel_gemv_fp32_q4_0_transb_bias_;

      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_input);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_weight);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_bias);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_output);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &K);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &N);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &has_bias);

      int local_size_0 = 128;  // Must be <= 256
      int gws_0 = N * local_size_0;
      int gws_1 = batch_count;

      global_size = cl::NDRange(gws_0, gws_1);
      local_size = cl::NDRange(local_size_0, 1);

    } else {
      kernel_wrapper = kernel_fp32_q4_0_transb_bias_;

      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_input);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_weight);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_bias);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_output);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &M);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &K);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &N);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &has_bias);

      int tile_size = 16;
      int gws_0 = (N + tile_size - 1) / tile_size * tile_size;
      int gws_1 = (M + tile_size - 1) / tile_size * tile_size;
      int gws_2 = batch_count;

      global_size = cl::NDRange(gws_0, gws_1, gws_2);
      local_size = cl::NDRange(tile_size, tile_size, 1);
    }
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Unsupported weight data type in LinearOp: {}", weight_.dtype());
  }

  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLLinearOp setArg failed: {}", ret); }

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_wrapper->get(), cl::NullRange, global_size, local_size);

  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute linear kernel, error code: {}", error);
  }
}

void OpenCLLinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isRedirect()) {
    outputs.emplace_back(inputs[1]);
    return;
  }

  MLLM_RT_ASSERT(options_.impl_type == aops::LinearImplTypes::kDefault || options_.impl_type == aops::LinearImplTypes::kGGUF);

  const auto& input = inputs[0];
  auto input_shape = input.shape();
  MLLM_RT_ASSERT(input_shape.size() >= 2);
  MLLM_RT_ASSERT_EQ(input_shape.back(), options_.in_channels);

  auto output_shape = input_shape;
  output_shape.back() = options_.out_channels;

  outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
}

}  // namespace mllm::opencl
