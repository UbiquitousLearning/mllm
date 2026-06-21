// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/opencl/ops/RMSNormOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLRMSNormOp::OpenCLRMSNormOp(const aops::RMSNormOpOptions& options) : aops::RMSNormOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  kernel_f32_q4_ = runtime->buildKernel("rmsnorm", "rmsnorm_f32_q4", {});
  kernel_f16_q4_ = runtime->buildKernel("rmsnorm", "rmsnorm_f16_q4", {});
}

void OpenCLRMSNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  // Should support [B, S, H ,D] and [B, S, H * D]
  auto x_shape = input.shape();
  int D = x_shape[x_shape.size() - 1];
  int other_dim_size = 1;
  for (size_t i = 0; i < x_shape.size() - 1; ++i) { other_dim_size *= x_shape[i]; }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_input = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_weight = (cl_mem)weight_.impl()->storage()->ptr_;
  auto cl_buffer_output = (cl_mem)output.impl()->storage()->ptr_;

  int weight_is_q4 = (weight_.dtype() == DataTypes::kGGUF_Q4_0) ? 1 : 0;
  int add_unit_offset = options_.add_unit_offset ? 1 : 0;
  float epsilon = options_.epsilon;

  std::shared_ptr<KernelWrap> kernel_wrapper;
  if (input.dtype() == DataTypes::kFloat32) {
    kernel_wrapper = kernel_f32_q4_;
  } else if (input.dtype() == DataTypes::kFloat16) {
    kernel_wrapper = kernel_f16_q4_;
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Unsupported input data type in RMSNormOp");
  }

  cl_int ret = CL_SUCCESS;
  ret |= kernel_wrapper->get().setArg(0, sizeof(cl_mem), &cl_buffer_input);
  ret |= kernel_wrapper->get().setArg(1, sizeof(cl_mem), &cl_buffer_output);
  ret |= kernel_wrapper->get().setArg(2, sizeof(cl_mem), &cl_buffer_weight);
  ret |= kernel_wrapper->get().setArg(3, sizeof(int), &weight_is_q4);
  ret |= kernel_wrapper->get().setArg(4, sizeof(int), &D);
  ret |= kernel_wrapper->get().setArg(5, sizeof(float), &epsilon);
  ret |= kernel_wrapper->get().setArg(6, sizeof(int), &add_unit_offset);

  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLRMSNormOp setArg failed: {}", ret); }

  // RMSNORM_WG_SIZE is 256
  int wg_size = 256;
  size_t global_size_0 = other_dim_size * wg_size;

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_wrapper->get(), cl::NullRange,
                                                            cl::NDRange(global_size_0, 1, 1), cl::NDRange(wg_size, 1, 1));

  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute rmsnorm kernel, error code: {}", error);
  }
}

}  // namespace mllm::opencl
