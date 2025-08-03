// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/SplitOp.hpp"

namespace mllm::cpu {

CPUSplitOp::CPUSplitOp(const aops::SplitOpOptions& options) : aops::SplitOp(options) {}

void CPUSplitOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  aops::SplitOp::forward(inputs, outputs);
}

}  // namespace mllm::cpu
