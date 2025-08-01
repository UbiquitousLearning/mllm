// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/PermuteOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPUPermuteOp::CPUPermuteOp(const aops::PermuteOpOptions& options) : aops::PermuteOp(options) {}

void CPUPermuteOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();

  switch (dtype) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::permute_fp32(input.ptr<mllm_fp32_t>(), output.ptr<mllm_fp32_t>(), input.shape().data(), options_.axis.data(),
                        options_.axis.size());
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::permute_fp16(input.ptr<mllm_fp16_t>(), output.ptr<mllm_fp16_t>(), input.shape().data(), options_.axis.data(),
                        options_.axis.size());
#endif
      break;
    }
    default: NYI("Data type not supported");
  }
}

}  // namespace mllm::cpu
