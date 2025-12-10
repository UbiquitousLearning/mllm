// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/backends/opencl/ops/TransposeOp.hpp"
#include "CL/cl_platform.h"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/utils/Common.hpp"
#include <numeric>

namespace mllm::opencl {

OpenCLTransposeOp::OpenCLTransposeOp(const aops::TransposeOpOptions& options) : aops::TransposeOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  kernel_generic_ = runtime->buildKernel("transpose", "transpose_generic", {});
  kernel_0213_ = runtime->buildKernel("transpose", "transpose_0213", {});
  kernel_0132_ = runtime->buildKernel("transpose", "transpose_0132", {});
}

void OpenCLTransposeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  int dim0 = options_.dim0;
  int dim1 = options_.dim1;
  int ndim = input.shape().size();

  if (dim0 < 0) dim0 += ndim;
  if (dim1 < 0) dim1 += ndim;

  std::vector<int> perm(ndim);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[dim0], perm[dim1]);

  auto cl_buffer_in = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_out = (cl_mem)output.impl()->storage()->ptr_;

  bool use_0213 = (ndim == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3);
  bool use_0132 = (ndim == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2);

  cl_int error;
  if (use_0213) {
    int B = input.shape()[0];
    int S = input.shape()[1];
    int H = input.shape()[2];
    int D = input.shape()[3];

    cl_int ret = CL_SUCCESS;
    ret |= kernel_0213_->get().setArg(0, sizeof(cl_mem), &cl_buffer_in);
    ret |= kernel_0213_->get().setArg(1, sizeof(cl_mem), &cl_buffer_out);
    ret |= kernel_0213_->get().setArg(2, sizeof(int), &B);
    ret |= kernel_0213_->get().setArg(3, sizeof(int), &S);
    ret |= kernel_0213_->get().setArg(4, sizeof(int), &H);
    ret |= kernel_0213_->get().setArg(5, sizeof(int), &D);

    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLTransposeOp setArg failed: {}", ret); }

    size_t global_size = input.numel();
    error = runtime->commandQueue().enqueueNDRangeKernel(kernel_0213_->get(), cl::NullRange, cl::NDRange(global_size),
                                                         cl::NullRange);
  } else if (use_0132) {
    int B = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];

    cl_int ret = CL_SUCCESS;
    ret |= kernel_0132_->get().setArg(0, sizeof(cl_mem), &cl_buffer_in);
    ret |= kernel_0132_->get().setArg(1, sizeof(cl_mem), &cl_buffer_out);
    ret |= kernel_0132_->get().setArg(2, sizeof(int), &B);
    ret |= kernel_0132_->get().setArg(3, sizeof(int), &C);
    ret |= kernel_0132_->get().setArg(4, sizeof(int), &H);
    ret |= kernel_0132_->get().setArg(5, sizeof(int), &W);

    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLTransposeOp setArg failed: {}", ret); }

    size_t global_size = input.numel();
    error = runtime->commandQueue().enqueueNDRangeKernel(kernel_0132_->get(), cl::NullRange, cl::NDRange(global_size),
                                                         cl::NullRange);
  } else {
    std::vector<int> in_strides(ndim);
    int stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      in_strides[i] = stride;
      stride *= input.shape()[i];
    }

    std::vector<int> out_shape = output.shape();

    cl_int err;
    cl::Buffer buf_in_strides(runtime->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim * sizeof(int),
                              in_strides.data(), &err);
    cl::Buffer buf_out_shape(runtime->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim * sizeof(int), out_shape.data(),
                             &err);
    cl::Buffer buf_perm(runtime->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim * sizeof(int), perm.data(), &err);

    int total_elements = input.numel();

    cl_int ret = CL_SUCCESS;
    ret |= kernel_generic_->get().setArg(0, sizeof(cl_mem), &cl_buffer_in);
    ret |= kernel_generic_->get().setArg(1, sizeof(cl_mem), &cl_buffer_out);
    ret |= kernel_generic_->get().setArg(2, sizeof(cl_mem), &buf_in_strides);
    ret |= kernel_generic_->get().setArg(3, sizeof(cl_mem), &buf_out_shape);
    ret |= kernel_generic_->get().setArg(4, sizeof(cl_mem), &buf_perm);
    ret |= kernel_generic_->get().setArg(5, sizeof(int), &ndim);
    ret |= kernel_generic_->get().setArg(6, sizeof(int), &total_elements);

    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLTransposeOp setArg failed: {}", ret); }

    error = runtime->commandQueue().enqueueNDRangeKernel(kernel_generic_->get(), cl::NullRange, cl::NDRange(total_elements),
                                                         cl::NullRange);
  }

  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute transpose kernel, error code: {}", error);
  }
}

}  // namespace mllm::opencl