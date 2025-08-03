// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/backends/cpu/ops/ViewOp.hpp"

namespace mllm::cpu {

CPUViewOp::CPUViewOp(const aops::ViewOpOptions& options) : aops::ViewOp(options) {}

void CPUViewOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  aops::ViewOp::forward(inputs, outputs);
}

}  // namespace mllm::cpu
