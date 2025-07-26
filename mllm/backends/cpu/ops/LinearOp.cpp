/**
 * @file LinearOp.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
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