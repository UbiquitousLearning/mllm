// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/SiLUOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

CPUSiLUOp::CPUSiLUOp(const aops::SiLUOpOptions& options) : aops::SiLUOp(options) {}

void CPUSiLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  switch (X.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      x86::silu_fp32(X.ptr<mllm_fp32_t>(), Y.ptr<mllm_fp32_t>(), X.numel(), options_.getThreads());
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::silu_fp32(X.ptr<mllm_fp32_t>(), Y.ptr<mllm_fp32_t>(), X.numel(), options_.getThreads());
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("SilU FP16 not implemented yet for X86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::silu_fp16(X.ptr<mllm_fp16_t>(), Y.ptr<mllm_fp16_t>(), X.numel(), options_.getThreads());
#endif
      break;
    }
    default: NYI("CPUSiLUOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
