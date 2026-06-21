// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/X2XOp.hpp"

namespace mllm::cpu {

CPUX2XOp::CPUX2XOp(const aops::X2XOpOptions& options) : aops::X2XOp(options) {}

void CPUX2XOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("Transfer CPU memory to other devices should be implemented in device backends");
}

}  // namespace mllm::cpu
