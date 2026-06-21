// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "ElewiseOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include <numeric>
#include <algorithm>

namespace mllm::opencl {

struct BroadcastInfo {
  bool can_be_broadcast_naive = false;
  int32_t broadcast_naive_loops = 0;
  int32_t batch_dims = 1;
  int32_t size = 0;
  int broadcast_input_idx = -1;  // 0 for input0, 1 for input1
};

BroadcastInfo calculateBroadcastInfo(const std::vector<int32_t>& a_shape, const std::vector<int32_t>& b_shape) {
  BroadcastInfo info;

  // Determine the broadcasted shape
  int a_rank = a_shape.size();
  int b_rank = b_shape.size();
  int max_rank = std::max(a_rank, b_rank);

  std::vector<int32_t> a_shape_padded(max_rank);
  std::vector<int32_t> b_shape_padded(max_rank);

  // Pad the shorter shape with 1s
  for (int i = 0; i < max_rank; ++i) {
    a_shape_padded[i] = (i < max_rank - a_rank) ? 1 : a_shape[i - (max_rank - a_rank)];
    b_shape_padded[i] = (i < max_rank - b_rank) ? 1 : b_shape[i - (max_rank - b_rank)];
  }

  // Compute the final broadcasted shape
  std::vector<int32_t> result_shape(max_rank);
  for (int i = 0; i < max_rank; ++i) { result_shape[i] = std::max(a_shape_padded[i], b_shape_padded[i]); }

  // Compute batch_dims
  int batch_dims = 0;
  for (int i = 0; i < max_rank; ++i) {
    if (a_shape_padded[i] == b_shape_padded[i]) {
      batch_dims++;
    } else {
      break;
    }
  }

  // Compute the total number of elements in the batch dimension
  int64_t batch_size = 1;
  for (int i = 0; i < batch_dims; ++i) { batch_size *= result_shape[i]; }
  info.batch_dims = static_cast<int32_t>(batch_size);

  // If both shapes are identical, no broadcasting is needed, return directly
  if (batch_dims == max_rank) {
    info.can_be_broadcast_naive = false;
    // size = all elements
    info.size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<>());
    return info;
  }

  // if can be broadcast naive
  int non_batch_rank = max_rank - batch_dims;
  std::vector<int32_t> a_non_batch_shape(a_shape_padded.begin() + batch_dims, a_shape_padded.end());
  std::vector<int32_t> b_non_batch_shape(b_shape_padded.begin() + batch_dims, b_shape_padded.end());
  std::vector<int32_t> result_non_batch_shape(result_shape.begin() + batch_dims, result_shape.end());

  // if a can be broadcast naive to result
  int first_diff_a = -1;
  for (int i = 0; i < non_batch_rank; ++i) {
    if (a_non_batch_shape[i] != result_non_batch_shape[i]) {
      first_diff_a = i;
      break;
    }
  }

  bool a_can_be_naive = false;
  if (first_diff_a != -1 && a_non_batch_shape[first_diff_a] == 1) {
    a_can_be_naive = true;
    for (int i = first_diff_a + 1; i < non_batch_rank; ++i) {
      if (a_non_batch_shape[i] != result_non_batch_shape[i]) {
        a_can_be_naive = false;
        break;
      }
    }
  }

  // if b can be broadcast naive to result
  int first_diff_b = -1;
  for (int i = 0; i < non_batch_rank; ++i) {
    if (b_non_batch_shape[i] != result_non_batch_shape[i]) {
      first_diff_b = i;
      break;
    }
  }

  bool b_can_be_naive = false;
  if (first_diff_b != -1 && b_non_batch_shape[first_diff_b] == 1) {
    b_can_be_naive = true;
    for (int i = first_diff_b + 1; i < non_batch_rank; ++i) {
      if (b_non_batch_shape[i] != result_non_batch_shape[i]) {
        b_can_be_naive = false;
        break;
      }
    }
  }

  // only enable this optimization when one tensor needs naive broadcast and the other does not
  if (a_can_be_naive && !b_can_be_naive) {
    info.can_be_broadcast_naive = true;
    info.broadcast_naive_loops = result_non_batch_shape[first_diff_a];
    info.broadcast_input_idx = 0;

    // size = non batch_elements / broadcast_naive_loops
    int non_batch_elements =
        std::accumulate(result_non_batch_shape.begin(), result_non_batch_shape.end(), 1, std::multiplies<>());
    info.size = non_batch_elements / info.broadcast_naive_loops;
  } else if (b_can_be_naive && !a_can_be_naive) {
    info.can_be_broadcast_naive = true;
    info.broadcast_naive_loops = result_non_batch_shape[first_diff_b];
    info.broadcast_input_idx = 1;

    int non_batch_elements =
        std::accumulate(result_non_batch_shape.begin(), result_non_batch_shape.end(), 1, std::multiplies<>());
    info.size = non_batch_elements / info.broadcast_naive_loops;
  } else {
    // cannot be broadcast naive
    info.size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<>());
  }

  return info;
}

OpenCLAddOp::OpenCLAddOp(const aops::AddOpOptions& options) : aops::AddOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("add", "add_float", {});
  kernel_broadcast_fp32_buffer_ = runtime->buildKernel("add", "add_broadcast_float", {});
}

void OpenCLAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input_a = inputs[0];
  auto& input_b = inputs[1];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_a = (cl_mem)input_a.impl()->storage()->ptr_;
  auto cl_buffer_b = (cl_mem)input_b.impl()->storage()->ptr_;
  auto cl_buffer_c = (cl_mem)output.impl()->storage()->ptr_;

  auto broadcast_info = calculateBroadcastInfo(input_a.shape(), input_b.shape());

  if (input_a.numel() == input_b.numel()) {
    cl_int ret = CL_SUCCESS;
    ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
    ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
    ret |= kernel_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLAddOp setArg failed: {}", ret); }

    size_t global_size = input_a.numel();

    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);

    if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute add kernel, error code: {}", error); }
  } else if (broadcast_info.can_be_broadcast_naive || input_a.numel() == 1 || input_b.numel() == 1) {
    int batch_size = broadcast_info.batch_dims;
    int loop_size = broadcast_info.broadcast_naive_loops;
    int vector_size = broadcast_info.size;
    int broadcast_input_idx = broadcast_info.broadcast_input_idx;

    if (input_a.numel() == 1) {
      batch_size = 1;
      loop_size = input_b.numel();
      vector_size = 1;
      broadcast_input_idx = 0;
    } else if (input_b.numel() == 1) {
      batch_size = 1;
      loop_size = input_a.numel();
      vector_size = 1;
      broadcast_input_idx = 1;
    }

    int batch_stride_a, loop_stride_a;
    int batch_stride_b, loop_stride_b;

    if (broadcast_input_idx == 0) {  // A is broadcasted
      batch_stride_a = vector_size;
      loop_stride_a = 0;

      batch_stride_b = loop_size * vector_size;
      loop_stride_b = vector_size;
    } else {  // B is broadcasted
      batch_stride_a = loop_size * vector_size;
      loop_stride_a = vector_size;

      batch_stride_b = vector_size;
      loop_stride_b = 0;
    }

    if (input_a.numel() == 1) {
      batch_stride_a = 0;
      loop_stride_a = 0;
      batch_stride_b = 1;
      loop_stride_b = 1;
    } else if (input_b.numel() == 1) {
      batch_stride_a = 1;
      loop_stride_a = 1;
      batch_stride_b = 0;
      loop_stride_b = 0;
    }

    cl_int ret = CL_SUCCESS;
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(3, sizeof(int), &batch_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(4, sizeof(int), &loop_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(5, sizeof(int), &vector_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(6, sizeof(int), &batch_stride_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(7, sizeof(int), &loop_stride_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(8, sizeof(int), &batch_stride_b);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(9, sizeof(int), &loop_stride_b);

    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLAddOp broadcast setArg failed: {}", ret); }

    size_t global_size = batch_size * loop_size * vector_size;
    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_broadcast_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute add broadcast kernel, error code: {}", error);
    }
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "AddOp broadcast not supported.");
  }
}

OpenCLSubOp::OpenCLSubOp(const aops::SubOpOptions& options) : aops::SubOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("sub", "sub_float", {});
  kernel_broadcast_fp32_buffer_ = runtime->buildKernel("sub", "sub_broadcast_float", {});
}

void OpenCLSubOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input_a = inputs[0];
  auto& input_b = inputs[1];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_a = (cl_mem)input_a.impl()->storage()->ptr_;
  auto cl_buffer_b = (cl_mem)input_b.impl()->storage()->ptr_;
  auto cl_buffer_c = (cl_mem)output.impl()->storage()->ptr_;

  auto broadcast_info = calculateBroadcastInfo(input_a.shape(), input_b.shape());

  if (input_a.numel() == input_b.numel()) {
    cl_int ret = CL_SUCCESS;
    ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
    ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
    ret |= kernel_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLSubOp setArg failed: {}", ret); }

    size_t global_size = input_a.numel();

    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);

    if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute sub kernel, error code: {}", error); }
  } else if (broadcast_info.can_be_broadcast_naive || input_a.numel() == 1 || input_b.numel() == 1) {
    int batch_size = broadcast_info.batch_dims;
    int loop_size = broadcast_info.broadcast_naive_loops;
    int vector_size = broadcast_info.size;
    int broadcast_input_idx = broadcast_info.broadcast_input_idx;

    if (input_a.numel() == 1) {
      batch_size = 1;
      loop_size = input_b.numel();
      vector_size = 1;
      broadcast_input_idx = 0;
    } else if (input_b.numel() == 1) {
      batch_size = 1;
      loop_size = input_a.numel();
      vector_size = 1;
      broadcast_input_idx = 1;
    }

    int batch_stride_a, loop_stride_a;
    int batch_stride_b, loop_stride_b;

    if (broadcast_input_idx == 0) {  // A is broadcasted
      batch_stride_a = vector_size;
      loop_stride_a = 0;

      batch_stride_b = loop_size * vector_size;
      loop_stride_b = vector_size;
    } else {  // B is broadcasted
      batch_stride_a = loop_size * vector_size;
      loop_stride_a = vector_size;

      batch_stride_b = vector_size;
      loop_stride_b = 0;
    }

    if (input_a.numel() == 1) {
      batch_stride_a = 0;
      loop_stride_a = 0;
      batch_stride_b = 1;
      loop_stride_b = 1;
    } else if (input_b.numel() == 1) {
      batch_stride_a = 1;
      loop_stride_a = 1;
      batch_stride_b = 0;
      loop_stride_b = 0;
    }

    cl_int ret = CL_SUCCESS;
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(3, sizeof(int), &batch_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(4, sizeof(int), &loop_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(5, sizeof(int), &vector_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(6, sizeof(int), &batch_stride_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(7, sizeof(int), &loop_stride_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(8, sizeof(int), &batch_stride_b);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(9, sizeof(int), &loop_stride_b);

    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLSubOp broadcast setArg failed: {}", ret); }

    size_t global_size = batch_size * loop_size * vector_size;
    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_broadcast_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute sub broadcast kernel, error code: {}", error);
    }
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "SubOp broadcast not supported.");
  }
}

OpenCLMulOp::OpenCLMulOp(const aops::MulOpOptions& options) : aops::MulOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("mul", "mul_float", {});
  kernel_broadcast_fp32_buffer_ = runtime->buildKernel("mul", "mul_broadcast_float", {});
}

void OpenCLMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input_a = inputs[0];
  auto& input_b = inputs[1];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_a = (cl_mem)input_a.impl()->storage()->ptr_;
  auto cl_buffer_b = (cl_mem)input_b.impl()->storage()->ptr_;
  auto cl_buffer_c = (cl_mem)output.impl()->storage()->ptr_;

  auto broadcast_info = calculateBroadcastInfo(input_a.shape(), input_b.shape());

  if (input_a.numel() == input_b.numel()) {
    cl_int ret = CL_SUCCESS;
    ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
    ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
    ret |= kernel_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLMulOp setArg failed: {}", ret); }

    size_t global_size = input_a.numel();

    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);

    if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute mul kernel, error code: {}", error); }
  } else if (broadcast_info.can_be_broadcast_naive || input_a.numel() == 1 || input_b.numel() == 1) {
    int batch_size = broadcast_info.batch_dims;
    int loop_size = broadcast_info.broadcast_naive_loops;
    int vector_size = broadcast_info.size;
    int broadcast_input_idx = broadcast_info.broadcast_input_idx;

    if (input_a.numel() == 1) {
      batch_size = 1;
      loop_size = input_b.numel();
      vector_size = 1;
      broadcast_input_idx = 0;
    } else if (input_b.numel() == 1) {
      batch_size = 1;
      loop_size = input_a.numel();
      vector_size = 1;
      broadcast_input_idx = 1;
    }

    int batch_stride_a, loop_stride_a;
    int batch_stride_b, loop_stride_b;

    if (broadcast_input_idx == 0) {  // A is broadcasted
      batch_stride_a = vector_size;
      loop_stride_a = 0;

      batch_stride_b = loop_size * vector_size;
      loop_stride_b = vector_size;
    } else {  // B is broadcasted
      batch_stride_a = loop_size * vector_size;
      loop_stride_a = vector_size;

      batch_stride_b = vector_size;
      loop_stride_b = 0;
    }

    if (input_a.numel() == 1) {
      batch_stride_a = 0;
      loop_stride_a = 0;
      batch_stride_b = 1;
      loop_stride_b = 1;
    } else if (input_b.numel() == 1) {
      batch_stride_a = 1;
      loop_stride_a = 1;
      batch_stride_b = 0;
      loop_stride_b = 0;
    }

    cl_int ret = CL_SUCCESS;
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(3, sizeof(int), &batch_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(4, sizeof(int), &loop_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(5, sizeof(int), &vector_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(6, sizeof(int), &batch_stride_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(7, sizeof(int), &loop_stride_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(8, sizeof(int), &batch_stride_b);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(9, sizeof(int), &loop_stride_b);

    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLMulOp broadcast setArg failed: {}", ret); }

    size_t global_size = batch_size * loop_size * vector_size;
    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_broadcast_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute mul broadcast kernel, error code: {}", error);
    }
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "MulOp broadcast not supported.");
  }
}

OpenCLDivOp::OpenCLDivOp(const aops::DivOpOptions& options) : aops::DivOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  kernel_fp32_buffer_ = runtime->buildKernel("div", "div_float", {});
  kernel_broadcast_fp32_buffer_ = runtime->buildKernel("div", "div_broadcast_float", {});
}

void OpenCLDivOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input_a = inputs[0];
  auto& input_b = inputs[1];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_a = (cl_mem)input_a.impl()->storage()->ptr_;
  auto cl_buffer_b = (cl_mem)input_b.impl()->storage()->ptr_;
  auto cl_buffer_c = (cl_mem)output.impl()->storage()->ptr_;

  auto broadcast_info = calculateBroadcastInfo(input_a.shape(), input_b.shape());

  if (input_a.numel() == input_b.numel()) {
    cl_int ret = CL_SUCCESS;
    ret |= kernel_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
    ret |= kernel_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
    ret |= kernel_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLDivOp setArg failed: {}", ret); }

    size_t global_size = input_a.numel();

    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);

    if (error != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute div kernel, error code: {}", error); }
  } else if (broadcast_info.can_be_broadcast_naive || input_a.numel() == 1 || input_b.numel() == 1) {
    int batch_size = broadcast_info.batch_dims;
    int loop_size = broadcast_info.broadcast_naive_loops;
    int vector_size = broadcast_info.size;
    int broadcast_input_idx = broadcast_info.broadcast_input_idx;

    if (input_a.numel() == 1) {
      batch_size = 1;
      loop_size = input_b.numel();
      vector_size = 1;
      broadcast_input_idx = 0;
    } else if (input_b.numel() == 1) {
      batch_size = 1;
      loop_size = input_a.numel();
      vector_size = 1;
      broadcast_input_idx = 1;
    }

    int batch_stride_a, loop_stride_a;
    int batch_stride_b, loop_stride_b;

    if (broadcast_input_idx == 0) {  // A is broadcasted
      batch_stride_a = vector_size;
      loop_stride_a = 0;

      batch_stride_b = loop_size * vector_size;
      loop_stride_b = vector_size;
    } else {  // B is broadcasted
      batch_stride_a = loop_size * vector_size;
      loop_stride_a = vector_size;

      batch_stride_b = vector_size;
      loop_stride_b = 0;
    }

    if (input_a.numel() == 1) {
      batch_stride_a = 0;
      loop_stride_a = 0;
      batch_stride_b = 1;
      loop_stride_b = 1;
    } else if (input_b.numel() == 1) {
      batch_stride_a = 1;
      loop_stride_a = 1;
      batch_stride_b = 0;
      loop_stride_b = 0;
    }

    cl_int ret = CL_SUCCESS;
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(0, sizeof(cl_mem), &cl_buffer_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(1, sizeof(cl_mem), &cl_buffer_b);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(2, sizeof(cl_mem), &cl_buffer_c);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(3, sizeof(int), &batch_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(4, sizeof(int), &loop_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(5, sizeof(int), &vector_size);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(6, sizeof(int), &batch_stride_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(7, sizeof(int), &loop_stride_a);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(8, sizeof(int), &batch_stride_b);
    ret |= kernel_broadcast_fp32_buffer_->get().setArg(9, sizeof(int), &loop_stride_b);

    if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLDivOp broadcast setArg failed: {}", ret); }

    size_t global_size = batch_size * loop_size * vector_size;
    auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_broadcast_fp32_buffer_->get(), cl::NullRange,
                                                              cl::NDRange(global_size), cl::NullRange);
    if (error != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute div broadcast kernel, error code: {}", error);
    }
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "DivOp broadcast not supported.");
  }
}

}  // namespace mllm::opencl