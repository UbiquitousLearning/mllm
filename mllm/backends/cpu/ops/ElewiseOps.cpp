// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/mllm.hpp"

namespace mllm::cpu {

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

    // size = non batch_elements / broadcast_naive_loops
    int non_batch_elements =
        std::accumulate(result_non_batch_shape.begin(), result_non_batch_shape.end(), 1, std::multiplies<>());
    info.size = non_batch_elements / info.broadcast_naive_loops;
  } else if (b_can_be_naive && !a_can_be_naive) {
    info.can_be_broadcast_naive = true;
    info.broadcast_naive_loops = result_non_batch_shape[first_diff_b];

    int non_batch_elements =
        std::accumulate(result_non_batch_shape.begin(), result_non_batch_shape.end(), 1, std::multiplies<>());
    info.size = non_batch_elements / info.broadcast_naive_loops;
  } else {
    // cannot be broadcast naive
    info.size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<>());
  }

  return info;
}

CPUAddOp::CPUAddOp(const aops::AddOpOptions& options) : aops::AddOp(options) {}

void CPUAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = output.dtype();

  auto broadcast_info = calculateBroadcastInfo(input0.shape(), input1.shape());
  bool can_be_broadcast_naive = broadcast_info.can_be_broadcast_naive;
  int32_t batch_dims = broadcast_info.batch_dims;
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t vector_size = broadcast_info.size;

  switch (dtype) {
    case kFloat32: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_fp32(output.ptr<mllm_fp32_t>(), input0.ptr<mllm_fp32_t>(), input1.ptr<mllm_fp32_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_fp32_scalar(output.ptr<mllm_fp32_t>(), input0.ptr<mllm_fp32_t>(), *input1.ptr<mllm_fp32_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else if (can_be_broadcast_naive) {
        const float* a = input0.ptr<mllm_fp32_t>();
        const float* b = input1.ptr<mllm_fp32_t>();
        float* out = output.ptr<mllm_fp32_t>();

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        // Process each batch separately
        for (int batch = 0; batch < batch_dims; ++batch) {
          // Each batch processes broadcast_naive_loops iterations of vector_size elements
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            int a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            int b_offset = batch * vector_size;  // b doesn't broadcast over loops dimension
            int out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;

            cpu::arm::ew_add_fp32(out + out_offset, a + a_offset, b + b_offset, vector_size, options_.getThreads());
          }
        }
#endif
      } else {
        NYI("AddOp broadcast not supported.");
      }
      break;
    }

    case kFloat16: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        cpu::arm::ew_add_fp16(output.ptr<mllm_fp16_t>(), input0.ptr<mllm_fp16_t>(), input1.ptr<mllm_fp16_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        cpu::arm::ew_add_fp16_scalar(output.ptr<mllm_fp16_t>(), input0.ptr<mllm_fp16_t>(), *input1.ptr<mllm_fp16_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else {
        NYI("AddOp broadcast not supported.");
      }
      break;
    }

    case kInt32: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_int32(output.ptr<mllm_int32_t>(), input0.ptr<mllm_int32_t>(), input1.ptr<mllm_int32_t>(),
                               output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_int32_scalar(output.ptr<mllm_int32_t>(), input0.ptr<mllm_int32_t>(), *input1.ptr<mllm_int32_t>(),
                                      output.numel(), options_.getThreads());
#endif
      } else {
        NYI("AddOp broadcast not supported.");
      }
      break;
    }

    case kInt16: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_int16(output.ptr<mllm_int16_t>(), input0.ptr<mllm_int16_t>(), input1.ptr<mllm_int16_t>(),
                               output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_int16_scalar(output.ptr<mllm_int16_t>(), input0.ptr<mllm_int16_t>(), *input1.ptr<mllm_int16_t>(),
                                      output.numel(), options_.getThreads());
#endif
      } else {
        NYI("AddOp broadcast not supported.");
      }
      break;
    }

    case kInt8: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_int8(output.ptr<mllm_int8_t>(), input0.ptr<mllm_int8_t>(), input1.ptr<mllm_int8_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_int8_scalar(output.ptr<mllm_int8_t>(), input0.ptr<mllm_int8_t>(), *input1.ptr<mllm_int8_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else {
        NYI("AddOp broadcast not supported.");
      }
      break;
    }

    case kComplexFloat32: {
      // currently only support scalar rhs
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_fp32_complex(output.ptr<mllm_complex_fp32_t>(), input0.ptr<mllm_fp32_t>(),
                                      input1.ptr<mllm_complex_fp32_t>(), output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_add_fp32_complex_scalar(output.ptr<mllm_complex_fp32_t>(), input0.ptr<mllm_fp32_t>(),
                                             *input1.ptr<mllm_complex_fp32_t>(), output.numel(), options_.getThreads());
#endif
      } else if (can_be_broadcast_naive) {
        const float* a = input0.ptr<mllm_fp32_t>();
        const mllm_complex_fp32_t* b = input1.ptr<mllm_complex_fp32_t>();
        mllm_complex_fp32_t* out = output.ptr<mllm_complex_fp32_t>();

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        // Process each batch separately
        for (int batch = 0; batch < batch_dims; ++batch) {
          // Each batch processes broadcast_naive_loops iterations of vector_size elements
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            int a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            int b_offset = batch * vector_size;  // b doesn't broadcast over loops dimension
            int out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;

            cpu::arm::ew_add_fp32_complex(out + out_offset, a + a_offset, b + b_offset, vector_size, options_.getThreads());
          }
        }
#endif
      } else {
        NYI("AddOp broadcast for complex output not supported.");
      }
      break;
    }

    default: NYI("AddOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUSubOp::CPUSubOp(const aops::SubOpOptions& options) : aops::SubOp(options) {}

void CPUSubOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = output.dtype();

  auto broadcast_info = calculateBroadcastInfo(input0.shape(), input1.shape());
  bool can_be_broadcast_naive = broadcast_info.can_be_broadcast_naive;
  int32_t batch_dims = broadcast_info.batch_dims;
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t vector_size = broadcast_info.size;

  switch (dtype) {
    case kFloat32: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_fp32(output.ptr<mllm_fp32_t>(), input0.ptr<mllm_fp32_t>(), input1.ptr<mllm_fp32_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_fp32_scalar(output.ptr<mllm_fp32_t>(), input0.ptr<mllm_fp32_t>(), *input1.ptr<mllm_fp32_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else if (can_be_broadcast_naive) {
        const float* a = input0.ptr<mllm_fp32_t>();
        const float* b = input1.ptr<mllm_fp32_t>();
        float* out = output.ptr<mllm_fp32_t>();

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        // Process each batch separately
        for (int batch = 0; batch < batch_dims; ++batch) {
          // Each batch processes broadcast_naive_loops iterations of vector_size elements
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            int a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            int b_offset = batch * vector_size;  // b doesn't broadcast over loops dimension
            int out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;

            cpu::arm::ew_sub_fp32(out + out_offset, a + a_offset, b + b_offset, vector_size, options_.getThreads());
          }
        }
#endif
      } else {
        NYI("SubOp broadcast not supported.");
      }
      break;
    }

    case kFloat16: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        cpu::arm::ew_sub_fp16(output.ptr<mllm_fp16_t>(), input0.ptr<mllm_fp16_t>(), input1.ptr<mllm_fp16_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        cpu::arm::ew_sub_fp16_scalar(output.ptr<mllm_fp16_t>(), input0.ptr<mllm_fp16_t>(), *input1.ptr<mllm_fp16_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else {
        NYI("SubOp broadcast not supported.");
      }
      break;
    }

    case kInt32: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_int32(output.ptr<mllm_int32_t>(), input0.ptr<mllm_int32_t>(), input1.ptr<mllm_int32_t>(),
                               output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_int32_scalar(output.ptr<mllm_int32_t>(), input0.ptr<mllm_int32_t>(), *input1.ptr<mllm_int32_t>(),
                                      output.numel(), options_.getThreads());
#endif
      } else {
        NYI("SubOp broadcast not supported.");
      }
      break;
    }

    case kInt16: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_int16(output.ptr<mllm_int16_t>(), input0.ptr<mllm_int16_t>(), input1.ptr<mllm_int16_t>(),
                               output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_int16_scalar(output.ptr<mllm_int16_t>(), input0.ptr<mllm_int16_t>(), *input1.ptr<mllm_int16_t>(),
                                      output.numel(), options_.getThreads());
#endif
      } else {
        NYI("SubOp broadcast not supported.");
      }
      break;
    }

    case kInt8: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_int8(output.ptr<mllm_int8_t>(), input0.ptr<mllm_int8_t>(), input1.ptr<mllm_int8_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_int8_scalar(output.ptr<mllm_int8_t>(), input0.ptr<mllm_int8_t>(), *input1.ptr<mllm_int8_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else {
        NYI("SubOp broadcast not supported.");
      }
      break;
    }

    case kComplexFloat32: {
      // currently only support scalar rhs
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_fp32_complex(output.ptr<mllm_complex_fp32_t>(), input0.ptr<mllm_fp32_t>(),
                                      input1.ptr<mllm_complex_fp32_t>(), output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_sub_fp32_complex_scalar(output.ptr<mllm_complex_fp32_t>(), input0.ptr<mllm_fp32_t>(),
                                             *input1.ptr<mllm_complex_fp32_t>(), output.numel(), options_.getThreads());
#endif
      } else if (can_be_broadcast_naive) {
        const float* a = input0.ptr<mllm_fp32_t>();
        const mllm_complex_fp32_t* b = input1.ptr<mllm_complex_fp32_t>();
        mllm_complex_fp32_t* out = output.ptr<mllm_complex_fp32_t>();

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        // Process each batch separately
        for (int batch = 0; batch < batch_dims; ++batch) {
          // Each batch processes broadcast_naive_loops iterations of vector_size elements
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            int a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            int b_offset = batch * vector_size;  // b doesn't broadcast over loops dimension
            int out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;

            cpu::arm::ew_sub_fp32_complex(out + out_offset, a + a_offset, b + b_offset, vector_size, options_.getThreads());
          }
        }
#endif
      } else {
        NYI("SubOp broadcast for complex output not supported.");
      }
      break;
    }

    default: NYI("SubOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUMulOp::CPUMulOp(const aops::MulOpOptions& options) : aops::MulOp(options) {}

void CPUMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = output.dtype();

  auto broadcast_info = calculateBroadcastInfo(input0.shape(), input1.shape());
  bool can_be_broadcast_naive = broadcast_info.can_be_broadcast_naive;
  int32_t batch_dims = broadcast_info.batch_dims;
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t vector_size = broadcast_info.size;

  switch (dtype) {
    case kFloat32: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_fp32(output.ptr<mllm_fp32_t>(), input0.ptr<mllm_fp32_t>(), input1.ptr<mllm_fp32_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_fp32_scalar(output.ptr<mllm_fp32_t>(), input0.ptr<mllm_fp32_t>(), *input1.ptr<mllm_fp32_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else if (can_be_broadcast_naive) {
        const float* a = input0.ptr<mllm_fp32_t>();
        const float* b = input1.ptr<mllm_fp32_t>();
        float* out = output.ptr<mllm_fp32_t>();

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        // Process each batch separately
        for (int batch = 0; batch < batch_dims; ++batch) {
          // Each batch processes broadcast_naive_loops iterations of vector_size elements
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            int a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            int b_offset = batch * vector_size;  // b doesn't broadcast over loops dimension
            int out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;

            cpu::arm::ew_mul_fp32(out + out_offset, a + a_offset, b + b_offset, vector_size, options_.getThreads());
          }
        }
#endif
      } else {
        NYI("MulOp broadcast not supported.");
      }
      break;
    }

    case kFloat16: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        cpu::arm::ew_mul_fp16(output.ptr<mllm_fp16_t>(), input0.ptr<mllm_fp16_t>(), input1.ptr<mllm_fp16_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        cpu::arm::ew_mul_fp16_scalar(output.ptr<mllm_fp16_t>(), input0.ptr<mllm_fp16_t>(), *input1.ptr<mllm_fp16_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else {
        NYI("MulOp broadcast not supported.");
      }
      break;
    }

    case kInt32: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_int32(output.ptr<mllm_int32_t>(), input0.ptr<mllm_int32_t>(), input1.ptr<mllm_int32_t>(),
                               output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_int32_scalar(output.ptr<mllm_int32_t>(), input0.ptr<mllm_int32_t>(), *input1.ptr<mllm_int32_t>(),
                                      output.numel(), options_.getThreads());
#endif
      } else {
        NYI("MulOp broadcast not supported.");
      }
      break;
    }

    case kInt16: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_int16(output.ptr<mllm_int16_t>(), input0.ptr<mllm_int16_t>(), input1.ptr<mllm_int16_t>(),
                               output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_int16_scalar(output.ptr<mllm_int16_t>(), input0.ptr<mllm_int16_t>(), *input1.ptr<mllm_int16_t>(),
                                      output.numel(), options_.getThreads());
#endif
      } else {
        NYI("MulOp broadcast not supported.");
      }
      break;
    }

    case kInt8: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_int8(output.ptr<mllm_int8_t>(), input0.ptr<mllm_int8_t>(), input1.ptr<mllm_int8_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_int8_scalar(output.ptr<mllm_int8_t>(), input0.ptr<mllm_int8_t>(), *input1.ptr<mllm_int8_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else {
        NYI("MulOp broadcast not supported.");
      }
      break;
    }

    case kComplexFloat32: {
      // currently only support scalar rhs
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_fp32_complex(output.ptr<mllm_complex_fp32_t>(), input0.ptr<mllm_fp32_t>(),
                                      input1.ptr<mllm_complex_fp32_t>(), output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_mul_fp32_complex_scalar(output.ptr<mllm_complex_fp32_t>(), input0.ptr<mllm_fp32_t>(),
                                             *input1.ptr<mllm_complex_fp32_t>(), output.numel(), options_.getThreads());
#endif
      } else if (can_be_broadcast_naive) {
        const float* a = input0.ptr<mllm_fp32_t>();
        const mllm_complex_fp32_t* b = input1.ptr<mllm_complex_fp32_t>();
        mllm_complex_fp32_t* out = output.ptr<mllm_complex_fp32_t>();

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        // Process each batch separately
        for (int batch = 0; batch < batch_dims; ++batch) {
          // Each batch processes broadcast_naive_loops iterations of vector_size elements
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            int a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            int b_offset = batch * vector_size;  // b doesn't broadcast over loops dimension
            int out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;

            cpu::arm::ew_mul_fp32_complex(out + out_offset, a + a_offset, b + b_offset, vector_size, options_.getThreads());
          }
        }
#endif
      } else {
        NYI("MulOp broadcast for complex output not supported.");
      }
      break;
    }

    default: NYI("MulOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUDivOp::CPUDivOp(const aops::DivOpOptions& options) : aops::DivOp(options) {}

void CPUDivOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = output.dtype();

  auto broadcast_info = calculateBroadcastInfo(input0.shape(), input1.shape());
  bool can_be_broadcast_naive = broadcast_info.can_be_broadcast_naive;
  int32_t batch_dims = broadcast_info.batch_dims;
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t vector_size = broadcast_info.size;

  switch (dtype) {
    case kFloat32: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_fp32(output.ptr<mllm_fp32_t>(), input0.ptr<mllm_fp32_t>(), input1.ptr<mllm_fp32_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_fp32_scalar(output.ptr<mllm_fp32_t>(), input0.ptr<mllm_fp32_t>(), *input1.ptr<mllm_fp32_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else if (can_be_broadcast_naive) {
        const float* a = input0.ptr<mllm_fp32_t>();
        const float* b = input1.ptr<mllm_fp32_t>();
        float* out = output.ptr<mllm_fp32_t>();

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        // Process each batch separately
        for (int batch = 0; batch < batch_dims; ++batch) {
          // Each batch processes broadcast_naive_loops iterations of vector_size elements
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            int a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            int b_offset = batch * vector_size;  // b doesn't broadcast over loops dimension
            int out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;

            cpu::arm::ew_div_fp32(out + out_offset, a + a_offset, b + b_offset, vector_size, options_.getThreads());
          }
        }
#endif
      } else {
        NYI("DivOp broadcast not supported.");
      }
      break;
    }

    case kFloat16: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        cpu::arm::ew_div_fp16(output.ptr<mllm_fp16_t>(), input0.ptr<mllm_fp16_t>(), input1.ptr<mllm_fp16_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        cpu::arm::ew_div_fp16_scalar(output.ptr<mllm_fp16_t>(), input0.ptr<mllm_fp16_t>(), *input1.ptr<mllm_fp16_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else {
        NYI("DivOp broadcast not supported.");
      }
      break;
    }

    case kInt32: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_int32(output.ptr<mllm_int32_t>(), input0.ptr<mllm_int32_t>(), input1.ptr<mllm_int32_t>(),
                               output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_int32_scalar(output.ptr<mllm_int32_t>(), input0.ptr<mllm_int32_t>(), *input1.ptr<mllm_int32_t>(),
                                      output.numel(), options_.getThreads());
#endif
      } else {
        NYI("DivOp broadcast not supported.");
      }
      break;
    }

    case kInt16: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_int16(output.ptr<mllm_int16_t>(), input0.ptr<mllm_int16_t>(), input1.ptr<mllm_int16_t>(),
                               output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_int16_scalar(output.ptr<mllm_int16_t>(), input0.ptr<mllm_int16_t>(), *input1.ptr<mllm_int16_t>(),
                                      output.numel(), options_.getThreads());
#endif
      } else {
        NYI("DivOp broadcast not supported.");
      }
      break;
    }

    case kInt8: {
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_int8(output.ptr<mllm_int8_t>(), input0.ptr<mllm_int8_t>(), input1.ptr<mllm_int8_t>(), output.numel(),
                              options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_int8_scalar(output.ptr<mllm_int8_t>(), input0.ptr<mllm_int8_t>(), *input1.ptr<mllm_int8_t>(),
                                     output.numel(), options_.getThreads());
#endif
      } else {
        NYI("DivOp broadcast not supported.");
      }
      break;
    }

    case kComplexFloat32: {
      // currently only support scalar rhs
      if (input0.numel() == input1.numel()) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_fp32_complex(output.ptr<mllm_complex_fp32_t>(), input0.ptr<mllm_fp32_t>(),
                                      input1.ptr<mllm_complex_fp32_t>(), output.numel(), options_.getThreads());
#endif
      } else if (input1.numel() == 1) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        cpu::arm::ew_div_fp32_complex_scalar(output.ptr<mllm_complex_fp32_t>(), input0.ptr<mllm_fp32_t>(),
                                             *input1.ptr<mllm_complex_fp32_t>(), output.numel(), options_.getThreads());
#endif
      } else if (can_be_broadcast_naive) {
        const float* a = input0.ptr<mllm_fp32_t>();
        const mllm_complex_fp32_t* b = input1.ptr<mllm_complex_fp32_t>();
        mllm_complex_fp32_t* out = output.ptr<mllm_complex_fp32_t>();

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        // Process each batch separately
        for (int batch = 0; batch < batch_dims; ++batch) {
          // Each batch processes broadcast_naive_loops iterations of vector_size elements
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            int a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            int b_offset = batch * vector_size;  // b doesn't broadcast over loops dimension
            int out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;

            cpu::arm::ew_div_fp32_complex(out + out_offset, a + a_offset, b + b_offset, vector_size, options_.getThreads());
          }
        }
#endif
      } else {
        NYI("DivOp broadcast for complex output not supported.");
      }
      break;
    }

    default: NYI("DivOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUNegOp::CPUNegOp(const aops::NegOpOptions& options) : aops::NegOp(options) {}

void CPUNegOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO:
  NYI("NegOp not support");
}

CPUAbsOp::CPUAbsOp(const aops::AbsOpOptions& options) : aops::AbsOp(options) {}

void CPUAbsOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();

  switch (dtype) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_abs_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }

    case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
      cpu::arm::ew_abs_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }

    case kInt32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_abs_int32(output.ptr<mllm_int32_t>(), input.ptr<mllm_int32_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }

    case kInt16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_abs_int16(output.ptr<mllm_int16_t>(), input.ptr<mllm_int16_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }

    case kInt8: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_abs_int8(output.ptr<mllm_int8_t>(), input.ptr<mllm_int8_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }

    case kComplexFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      // Just use simple std::abs()
      cpu::arm::ew_abs_complex_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_complex_fp32_t>(), output.numel(),
                                    options_.getThreads());
#endif
      break;
    }

    case kComplexFloat64: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      // Just use simple std::abs()
      cpu::arm::ew_abs_complex_fp64(output.ptr<mllm_fp32_t>(), input.ptr<mllm_complex_fp64_t>(), output.numel(),
                                    options_.getThreads());
#endif
      break;
    }

    default: NYI("AbsOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPULogOp::CPULogOp(const aops::LogOpOptions& options) : aops::LogOp(options) {}

void CPULogOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = output.dtype();

  switch (dtype) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_log_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_log_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }
#endif

    default: NYI("LogOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUExpOp::CPUExpOp(const aops::ExpOpOptions& options) : aops::ExpOp(options) {}

void CPUExpOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = output.dtype();
  switch (dtype) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_exp_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_exp_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }
    default: NYI("ExpOp dtype not supported.");
  }
}

CPUClipOp::CPUClipOp(const aops::ClipOpOptions& options) : aops::ClipOp(options) {}

void CPUClipOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();

  switch (dtype) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::clip_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), static_cast<mllm_fp32_t>(options_.min_val),
                          static_cast<mllm_fp32_t>(options_.max_val), output.numel(), options_.getThreads());
#endif
      break;
    }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::clip_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), static_cast<mllm_fp16_t>(options_.min_val),
                          static_cast<mllm_fp16_t>(options_.max_val), output.numel(), options_.getThreads());
#endif
      break;
    }
#endif

    case kInt8: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::clip_int8(output.ptr<mllm_int8_t>(), input.ptr<mllm_int8_t>(), static_cast<mllm_int8_t>(options_.min_val),
                          static_cast<mllm_int8_t>(options_.max_val), output.numel(), options_.getThreads());
#endif
      break;
    }

    case kInt16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::clip_int16(output.ptr<mllm_int16_t>(), input.ptr<mllm_int16_t>(), static_cast<mllm_int16_t>(options_.min_val),
                           static_cast<mllm_int16_t>(options_.max_val), output.numel(), options_.getThreads());
#endif
      break;
    }

    case kInt32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::clip_int32(output.ptr<mllm_int32_t>(), input.ptr<mllm_int32_t>(), static_cast<mllm_int32_t>(options_.min_val),
                           static_cast<mllm_int32_t>(options_.max_val), output.numel(), options_.getThreads());
#endif
      break;
    }

    default: NYI("ClipOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUSinOp::CPUSinOp(const aops::SinOpOptions& options) : aops::SinOp(options) {}

void CPUSinOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = output.dtype();
  switch (dtype) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_sin_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_sin_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }
    default: NYI("SinOp dtype not supported.");
  }
}

CPUCosOp::CPUCosOp(const aops::CosOpOptions& options) : aops::CosOp(options) {}

void CPUCosOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = output.dtype();
  switch (dtype) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_cos_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::arm::ew_cos_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), output.numel(), options_.getThreads());
#endif
      break;
    }
    default: NYI("CosOp dtype not supported.");
  }
}

}  // namespace mllm::cpu
