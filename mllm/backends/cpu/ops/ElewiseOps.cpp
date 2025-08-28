// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/mllm.hpp"

namespace mllm::cpu {

BroadcastInfo calculateBroadcastInfo(const std::vector<int32_t>& a_shape, const std::vector<int32_t>& b_shape) {
  BroadcastInfo info;
  int a_ndim = a_shape.size();
  int b_ndim = b_shape.size();

  //  Rules(can_be_broadcast_naive):
  //  1. b must have fewer/equal dimensions than a (treat a as the "large" tensor).
  //  2. The last k dimensions of b must exactly match the last k dimensions of a,
  //     where k == b_ndim.
  //  3. All leading dimensions of b must be 1 so that a simple stride broadcast works.
  {
    if (b_ndim <= a_ndim) {
      bool can_broadcast = true;
      for (int i = 1; i <= b_ndim; ++i) {
        int a_idx = a_ndim - i;
        int b_idx = b_ndim - i;
        if (a_shape[a_idx] != b_shape[b_idx] && a_shape[a_idx] != 1 && b_shape[b_idx] != 1) {
          can_broadcast = false;
          break;
        }
      }
      info.can_be_broadcast_naive = can_broadcast;
    }

    if (info.can_be_broadcast_naive) {
      info.broadcast_naive_loops = 1;
      for (int i = 0; i < a_ndim; ++i) {
        if (i < a_ndim - b_ndim) {
          info.broadcast_naive_loops *= a_shape[i];
        } else {
          int b_idx = i - (a_ndim - b_ndim);
          if (b_shape[b_idx] == 1) { info.broadcast_naive_loops *= a_shape[i]; }
        }
      }

      info.broadcast_naive_stride = 1;
      for (int i = 0; i < b_ndim; ++i) { info.broadcast_naive_stride *= b_shape[i]; }
    }
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
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t broadcast_naive_stride = broadcast_info.broadcast_naive_stride;

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
        // Each iteration processes one contiguous block of size `stride`
        for (int l = 0; l < broadcast_naive_loops; ++l) {
          cpu::arm::ew_add_fp32(out + l * broadcast_naive_stride, a + l * broadcast_naive_stride,
                                b,  // b always contains `stride` elements
                                broadcast_naive_stride, options_.getThreads());
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
        // Each iteration processes one contiguous block of size `stride`
        for (int l = 0; l < broadcast_naive_loops; ++l) {
          cpu::arm::ew_add_fp32_complex(out + l * broadcast_naive_stride, a + l * broadcast_naive_stride,
                                        b,  // b always contains `stride` elements
                                        broadcast_naive_stride, options_.getThreads());
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
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t broadcast_naive_stride = broadcast_info.broadcast_naive_stride;

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
        // Each iteration processes one contiguous block of size `stride`
        for (int l = 0; l < broadcast_naive_loops; ++l) {
          cpu::arm::ew_sub_fp32(out + l * broadcast_naive_stride, a + l * broadcast_naive_stride,
                                b,  // b always contains `stride` elements
                                broadcast_naive_stride, options_.getThreads());
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
        // Each iteration processes one contiguous block of size `stride`
        for (int l = 0; l < broadcast_naive_loops; ++l) {
          cpu::arm::ew_sub_fp32_complex(out + l * broadcast_naive_stride, a + l * broadcast_naive_stride,
                                        b,  // b always contains `stride` elements
                                        broadcast_naive_stride, options_.getThreads());
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
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t broadcast_naive_stride = broadcast_info.broadcast_naive_stride;

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
        // Each iteration processes one contiguous block of size `stride`
        for (int l = 0; l < broadcast_naive_loops; ++l) {
          cpu::arm::ew_mul_fp32(out + l * broadcast_naive_stride, a + l * broadcast_naive_stride,
                                b,  // b always contains `stride` elements
                                broadcast_naive_stride, options_.getThreads());
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
        // Each iteration processes one contiguous block of size `stride`
        for (int l = 0; l < broadcast_naive_loops; ++l) {
          cpu::arm::ew_mul_fp32_complex(out + l * broadcast_naive_stride, a + l * broadcast_naive_stride,
                                        b,  // b always contains `stride` elements
                                        broadcast_naive_stride, options_.getThreads());
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
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t broadcast_naive_stride = broadcast_info.broadcast_naive_stride;

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
        // Each iteration processes one contiguous block of size `stride`
        for (int l = 0; l < broadcast_naive_loops; ++l) {
          cpu::arm::ew_div_fp32(out + l * broadcast_naive_stride, a + l * broadcast_naive_stride,
                                b,  // b always contains `stride` elements
                                broadcast_naive_stride, options_.getThreads());
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
        // Each iteration processes one contiguous block of size `stride`
        for (int l = 0; l < broadcast_naive_loops; ++l) {
          cpu::arm::ew_div_fp32_complex(out + l * broadcast_naive_stride, a + l * broadcast_naive_stride,
                                        b,  // b always contains `stride` elements
                                        broadcast_naive_stride, options_.getThreads());
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
