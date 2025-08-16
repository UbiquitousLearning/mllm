// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/RMSNormOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPURMSNormOp::CPURMSNormOp(const aops::RMSNormOpOptions& options) : aops::RMSNormOp(options) {}

void CPURMSNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& i = inputs[0];
  auto& o = outputs[0];

  // Should support [B, S, H ,D] and [B, S, H * D]
  auto x_shape = i.shape();
  auto D = x_shape[x_shape.size() - 1];
  size_t other_dim_size = 1;
  for (size_t i = 0; i < x_shape.size() - 1; ++i) { other_dim_size *= x_shape[i]; }

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(i.isContiguous());

  switch (i.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#pragma omp parallel for num_threads(options_.getThreads()) if (options_.getThreads() > 1)
      for (size_t other_dim = 0; other_dim < other_dim_size; ++other_dim) {
        arm::rmsnorm_fp32(i.ptr<mllm_fp32_t>() + other_dim * D, weight_.ptr<mllm_fp32_t>(),
                          o.ptr<mllm_fp32_t>() + other_dim * D, D, options_.epsilon, options_.add_unit_offset,
                          options_.getThreads());
      }
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#pragma omp parallel for num_threads(options_.getThreads()) if (options_.getThreads() > 1)
      for (size_t other_dim = 0; other_dim < other_dim_size; ++other_dim) {
        arm::rmsnorm_fp16(i.ptr<mllm_fp16_t>() + other_dim * D, weight_.ptr<mllm_fp16_t>(),
                          o.ptr<mllm_fp16_t>() + other_dim * D, D, options_.epsilon, options_.add_unit_offset,
                          options_.getThreads());
      }
#endif
      break;
    }
    default: NYI("Unsupported data type");
  }
}

}  // namespace mllm::cpu
