// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/GELUOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

CPUGELUOp::CPUGELUOp(const aops::GELUOpOptions& options) : aops::GELUOp(options) {}

void CPUGELUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(X.isContiguous());

  switch (X.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      x86::gelu_fp32(Y.ptr<mllm_fp32_t>(), X.ptr<mllm_fp32_t>(), X.numel(), options_.getThreads());
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::gelu_fp32(Y.ptr<mllm_fp32_t>(), X.ptr<mllm_fp32_t>(), X.numel(), options_.getThreads());
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("CPUGELUOp::forward not support dtype {}", nameOfType(X.dtype()));
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::gelu_fp16(Y.ptr<mllm_fp16_t>(), X.ptr<mllm_fp16_t>(), X.numel(), options_.getThreads());
#endif
      break;
    }
    default: NYI("CPUGELUOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
