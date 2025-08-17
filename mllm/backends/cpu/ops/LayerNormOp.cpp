// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/LayerNormOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/Parallel.hpp"

namespace mllm::cpu {

CPULayerNormOp::CPULayerNormOp(const aops::LayerNormOpOptions& options) : aops::LayerNormOp(options) {}

void CPULayerNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& i = inputs[0];
  auto& o = outputs[0];

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(i.isContiguous());

  int32_t loop_size = 1;
  for (int i : options_.normalized_shape) { loop_size *= i; }

  MLLM_RT_ASSERT(loop_size > 0);
  MLLM_RT_ASSERT_EQ(i.numel() % loop_size, 0);

  // Calculate loop times
  size_t loop_times = i.numel() / loop_size;

  switch (i.dtype()) {
    case kFloat32: {
      MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, options_.getThreads(), l, 0, loop_times, 1, {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::layernorm_N_fp32(o.ptr<mllm_fp32_t>() + l * loop_size, i.ptr<mllm_fp32_t>() + l * loop_size,
                              options_.elementwise_affine ? weight_.ptr<mllm_fp32_t>() : nullptr,
                              options_.bias ? bias_.ptr<mllm_fp32_t>() : nullptr, loop_size, options_.eps,
                              options_.getThreads());
#endif
      });
      break;
    }
    case kFloat16: {
      MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, options_.getThreads(), l, 0, loop_times, 1, {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::layernorm_N_fp16(o.ptr<mllm_fp16_t>() + l * loop_size, i.ptr<mllm_fp16_t>() + l * loop_size,
                              options_.elementwise_affine ? weight_.ptr<mllm_fp16_t>() : nullptr,
                              options_.bias ? bias_.ptr<mllm_fp16_t>() : nullptr, loop_size, options_.eps,
                              options_.getThreads());
#endif
      });
      break;
    }
    default: NYI("CPULayerNormOp::forward not support dtype {}", nameOfType(i.dtype())); break;
  }
}

}  // namespace mllm::cpu
