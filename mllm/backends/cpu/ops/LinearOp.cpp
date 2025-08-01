// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/LinearOp.hpp"

namespace mllm::cpu {

CPULinearOp::CPULinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void CPULinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
  NYI("You find me, please implement me!");
}

void CPULinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
  LinearOp::reshape(inputs, outputs);
}
}  // namespace mllm::cpu