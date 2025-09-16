// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/ops/ReLUOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

CPUReLUOp::CPUReLUOp(const aops::ReLUOpOptions& options) : aops::ReLUOp(options) {}

void CPUReLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();
  switch (dtype) {
    case kFloat32: {
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto output_ptr = output.ptr<mllm_fp32_t>();
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::relu_fp32(input_ptr, output_ptr, input.numel(), options_.getThreads());
#else
      NYI("ReLU not supported for Other Architectures instead of ARM64");
#endif
      break;
    }
    case kFloat16: {
      auto input_ptr = input.ptr<mllm_fp16_t>();
      auto output_ptr = output.ptr<mllm_fp16_t>();
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::relu_fp16(input_ptr, output_ptr, input.numel(), options_.getThreads());
#else
      NYI("ReLU not supported for Other Architectures instead of ARM64");
#endif
      break;
    }
    default: NYI("ReLU not supported for data type: {}", nameOfType(dtype));
  }
}

}  // namespace mllm::cpu
