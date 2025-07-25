/**
 * @file LinearOp.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#include "mllm/backends/x86/ops/LinearOp.hpp"

namespace mllm::x86 {

X86LinearOp::X86LinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void X86LinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
  LinearOp::forward(inputs, outputs);
}

void X86LinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
  LinearOp::reshape(inputs, outputs);
}
}  // namespace mllm::x86