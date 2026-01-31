// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>

#include "mllm/backends/cpu/ops/TanhOp.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPUTanhOp::CPUTanhOp(const aops::TanhOpOptions& options) : aops::TanhOp(options) {}

void CPUTanhOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  const auto numel = X.numel();

  switch (X.dtype()) {
    case kFloat32: {
      const auto* x_ptr = X.ptr<mllm_fp32_t>();
      auto* y_ptr = Y.ptr<mllm_fp32_t>();
      MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, 4, idx, 0, numel, 1, {
        y_ptr[idx] = std::tanh(x_ptr[idx]);
      });
      break;
    }
    case kFloat16: {
      const auto* x_ptr = X.ptr<mllm_fp16_t>();
      auto* y_ptr = Y.ptr<mllm_fp16_t>();
      MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, 4, idx, 0, numel, 1, {
        float v = static_cast<float>(x_ptr[idx]);
        y_ptr[idx] = static_cast<mllm_fp16_t>(std::tanh(v));
      });
      break;
    }
    default: NYI("CPUTanhOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
