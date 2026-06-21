// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendViewOp.hpp"

namespace mllm::ascend {

AscendViewOp::AscendViewOp(const aops::ViewOpOptions& options) : aops::ViewOp(options) {}

void AscendViewOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // View operation only changes metadata (shape), not actual data
  // Just call the base class implementation which is empty
  aops::ViewOp::forward(inputs, outputs);
}

}  // namespace mllm::ascend