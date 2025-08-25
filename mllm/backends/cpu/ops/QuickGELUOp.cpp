// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/QuickGELUOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

CPUQuickGELUOp::CPUQuickGELUOp(const aops::QuickGELUOpOptions& options) : aops::QuickGELUOp(options) {}

void CPUQuickGELUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(X.isContiguous());

  switch (X.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      x86::quick_gelu_fp32(Y.ptr<mllm_fp32_t>(), X.ptr<mllm_fp32_t>(), X.numel(), options_.getThreads());
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::quick_gelu_fp32(Y.ptr<mllm_fp32_t>(), X.ptr<mllm_fp32_t>(), X.numel(), options_.getThreads());
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("CPUQuickGELUOp::forward not support dtype {}", nameOfType(X.dtype()));
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::quick_gelu_fp16(Y.ptr<mllm_fp16_t>(), X.ptr<mllm_fp16_t>(), X.numel(), options_.getThreads());
#endif
      break;
    }
    default: NYI("CPUQuickGELUOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
