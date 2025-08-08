// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/SoftmaxOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

CPUSoftmaxOp::CPUSoftmaxOp(const aops::SoftmaxOpOptions& options) : aops::SoftmaxOp(options) {}

void CPUSoftmaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto X = inputs[0];
  auto Y = outputs[0];

  MLLM_RT_ASSERT_EQ(X.shape().size(), 4);
  MLLM_RT_ASSERT(options_.axis == -1 || options_.axis == 3);

  auto B = X.shape()[0];
  auto H = X.shape()[1];
  auto S = X.shape()[2];
  auto D = X.shape()[3];

  switch (X.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      for (int b = 0; b < B; ++b) {
#pragma omp parallel for schedule(auto) num_threads(options_.getThreads()) if (options_.getThreads() > 1)
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            arm::softmax_v1_fp32(X.ptrAt<mllm_fp32_t>({b, h, s, 0}), Y.ptrAt<mllm_fp32_t>({b, h, s, 0}), D, 1,
                                 options_.getThreads());
          }
        }
      }
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      for (int b = 0; b < B; ++b) {
#pragma omp parallel for schedule(auto) num_threads(options_.getThreads()) if (options_.getThreads() > 1)
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            arm::softmax_v1_fp16(X.ptrAt<mllm_fp16_t>({b, h, s, 0}), Y.ptrAt<mllm_fp16_t>({b, h, s, 0}), D, 1,
                                 options_.getThreads());
          }
        }
      }
#endif
      break;
    }
    default: NYI("CPUSoftmaxOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
