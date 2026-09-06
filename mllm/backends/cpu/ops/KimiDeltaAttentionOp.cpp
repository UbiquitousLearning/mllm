// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/KimiDeltaAttentionOp.hpp"

#include <cstring>
#include <stdexcept>

#include "mllm/backends/cpu/kernels/common/kda/kimi_delta_attention.hpp"

namespace mllm::cpu {

CPUKimiDeltaAttentionOp::CPUKimiDeltaAttentionOp(const aops::KimiDeltaAttentionOpOptions& options)
    : aops::KimiDeltaAttentionOp(options) {}

void CPUKimiDeltaAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  for (const auto& input : inputs) {
    if (!input.isContiguous()) { throw std::invalid_argument("KimiDeltaAttention CPU inputs must be contiguous"); }
  }

  const auto& q = inputs[0];
  auto& output = outputs[0];
  auto& updated_state = outputs[1];
  if (!options_.state_inplace) { std::memcpy(updated_state.ptr<float>(), inputs[7].ptr<float>(), inputs[7].bytes()); }
  kda::kimiDeltaAttentionF32(inputs[0].ptr<float>(), inputs[1].ptr<float>(), inputs[2].ptr<float>(), inputs[3].ptr<float>(),
                             inputs[4].ptr<float>(), inputs[5].ptr<float>(), inputs[6].ptr<float>(), updated_state.ptr<float>(),
                             output.ptr<float>(), q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3], options_.safe_gate,
                             options_.lower_bound, options_.getThreads());
}

}  // namespace mllm::cpu
