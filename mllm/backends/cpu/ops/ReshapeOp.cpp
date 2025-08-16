// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/utils/Common.hpp"
#include "mllm/backends/cpu/ops/ReshapeOp.hpp"

namespace mllm::cpu {

CPUReshapeOp::CPUReshapeOp(const aops::ReshapeOpOptions& options) : aops::ReshapeOp(options) {}

void CPUReshapeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
  NYI("Pls use view instead. reshape is not implemented. If you want to reshape a un-contiguous tensor, please implement me.");
}

}  // namespace mllm::cpu
