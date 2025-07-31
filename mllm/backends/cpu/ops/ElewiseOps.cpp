/**
 * @file ElewiseOps.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 */
#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPUAddOp::CPUAddOp(const aops::AddOpOptions& options) : aops::AddOp(options) {}

void CPUAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = output.dtype();

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

    default: NYI("AddOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUSubOp::CPUSubOp(const aops::SubOpOptions& options) : aops::SubOp(options) {}

void CPUSubOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = output.dtype();

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
      } else {
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
      }
      break;
    }

    default: NYI("AddOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUMulOp::CPUMulOp(const aops::MulOpOptions& options) : aops::MulOp(options) {}

void CPUMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = output.dtype();

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
      } else {
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
      }
      break;
    }

    default: NYI("AddOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUDivOp::CPUDivOp(const aops::DivOpOptions& options) : aops::DivOp(options) {}

void CPUDivOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = output.dtype();

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
      } else {
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
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
        NYI("AddOp broadcast not supported.");
      }
      break;
    }

    default: NYI("AddOp not support data type: {}", nameOfType(dtype)); break;
  }
}

CPUNegOp::CPUNegOp(const aops::NegOpOptions& options) : aops::NegOp(options) {}

void CPUNegOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO:
  NYI("NegOp not support");
}

}  // namespace mllm::cpu