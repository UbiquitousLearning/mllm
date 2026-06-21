// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/ops/MatMulOp.hpp"
#include "CL/cl.h"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLMatMulOp::OpenCLMatMulOp(const aops::MatMulOpOptions& options) : aops::MatMulOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto transpose_a = options_.transpose_a;
  auto transpose_b = options_.transpose_b;

  std::string kernelName;
  if (!transpose_a && !transpose_b) {
    kernelName = "matmul_buffer_nt_nt_opt";
  } else if (!transpose_a && transpose_b) {
    kernelName = "matmul_buffer_nt_t_opt";
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "MatMulOp: Unsupported transpose combination: A={}, B={}", transpose_a,
                    transpose_b);
  }

  std::set<std::string> buildOptions;
  kernel_fp32_buffer_ = runtime->buildKernel("matmul", kernelName, buildOptions);
}

void OpenCLMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& lhs = inputs[0];
  auto& rhs = inputs[1];
  auto& o = outputs[0];

  auto transpose_a = options_.transpose_a;
  auto transpose_b = options_.transpose_b;

  auto lhs_shape = lhs.shape();
  auto rhs_shape = rhs.shape();

  MLLM_RT_ASSERT(lhs_shape.size() >= 2);
  MLLM_RT_ASSERT(rhs_shape.size() >= 2);

  const int lhs_rows = lhs_shape[lhs_shape.size() - 2];
  const int lhs_cols = lhs_shape[lhs_shape.size() - 1];
  const int rhs_rows = rhs_shape[rhs_shape.size() - 2];
  const int rhs_cols = rhs_shape[rhs_shape.size() - 1];
  const int M = transpose_a ? lhs_cols : lhs_rows;
  const int N = transpose_b ? rhs_rows : rhs_cols;
  const int K = transpose_a ? lhs_rows : lhs_cols;
  const int K_from_rhs = transpose_b ? rhs_cols : rhs_rows;
  MLLM_RT_ASSERT_EQ(K, K_from_rhs);

  int batch_count = 1;
  for (size_t i = 0; i < lhs_shape.size() - 2; ++i) { batch_count *= lhs_shape[i]; }
  int rhs_batch_count = 1;
  for (size_t i = 0; i < rhs_shape.size() - 2; ++i) { rhs_batch_count *= rhs_shape[i]; }
  MLLM_RT_ASSERT_EQ(batch_count, rhs_batch_count);

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  int A_batch_stride = 0;
  int B_batch_stride = 0;
  int C_batch_stride = M * N;

  if (batch_count > 1) {
    if (lhs_shape.size() > 2) { A_batch_stride = lhs.stride()[lhs_shape.size() - 3]; }
    if (rhs_shape.size() > 2) { B_batch_stride = rhs.stride()[rhs_shape.size() - 3]; }
    if (o.shape().size() > 2) { C_batch_stride = o.stride()[o.shape().size() - 3]; }
  }

  auto lhs_buffer = (cl_mem)lhs.impl()->ptr<void>();
  auto rhs_buffer = (cl_mem)rhs.impl()->ptr<void>();
  auto o_buffer = (cl_mem)o.impl()->ptr<void>();

  auto& cl_kernel = kernel_fp32_buffer_->get();
  cl_int err = 0;
  err |= cl_kernel.setArg(0, sizeof(int), &M);
  err |= cl_kernel.setArg(1, sizeof(int), &N);
  err |= cl_kernel.setArg(2, sizeof(int), &K);
  err |= cl_kernel.setArg(3, sizeof(cl_mem), &lhs_buffer);
  err |= cl_kernel.setArg(4, A_batch_stride);
  err |= cl_kernel.setArg(5, sizeof(cl_mem), &rhs_buffer);
  err |= cl_kernel.setArg(6, B_batch_stride);
  err |= cl_kernel.setArg(7, sizeof(cl_mem), &o_buffer);
  err |= cl_kernel.setArg(8, C_batch_stride);

  if (err != CL_SUCCESS) { MLLM_ERROR("Failed to set OpenCL MatMulOp kernel arguments, error code: {}", err); }

  const int TILE_SIZE = 16;
  cl::NDRange global((M + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE, batch_count);
  cl::NDRange local(TILE_SIZE, TILE_SIZE, 1);

  auto error = runtime->commandQueue().enqueueNDRangeKernel(cl_kernel, cl::NullRange, global, local);
  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to enqueue OpenCL MatMulOp kernel, error code: {}", error);
  }
}

}  // namespace mllm::opencl
