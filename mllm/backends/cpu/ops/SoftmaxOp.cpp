// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/SoftmaxOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/Parallel.hpp"

namespace mllm::cpu {

CPUSoftmaxOp::CPUSoftmaxOp(const aops::SoftmaxOpOptions& options) : aops::SoftmaxOp(options) {}

void CPUSoftmaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  const auto& Y = outputs[0];

  MLLM_RT_ASSERT_EQ(X.shape().size(), 4);
  MLLM_RT_ASSERT(options_.axis == -1 || options_.axis == 3);

  auto B = X.shape()[0];
  auto H = X.shape()[1];
  auto S = X.shape()[2];
  auto D = X.shape()[3];

  switch (X.dtype()) {
    case kFloat32: {
      for (int b = 0; b < B; ++b) {
        MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, options_.getThreads(), h, 0, H, 1, {
          for (int s = 0; s < S; ++s) {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
            // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
            arm::softmax_v1_fp32(X.cptrAt<mllm_fp32_t>({b, (int)h, s, 0}), Y.cptrAt<mllm_fp32_t>({b, (int)h, s, 0}), D, 1,
                                 options_.getThreads());
#endif
          }
        });
      }
      break;
    }
    case kFloat16: {
      for (int b = 0; b < B; ++b) {
        MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, options_.getThreads(), h, 0, H, 1, {
          for (int s = 0; s < S; ++s) {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
            // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
            arm::softmax_v1_fp16(X.cptrAt<mllm_fp16_t>({b, (int)h, s, 0}), Y.cptrAt<mllm_fp16_t>({b, (int)h, s, 0}), D, 1,
                                 options_.getThreads());
#endif
          }
        });
      }
      break;
    }
    default: NYI("CPUSoftmaxOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
