// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp"
#include "mllm/compile/passes/Pass.hpp"

namespace mllm::qnn {

std::vector<std::shared_ptr<ir::Pass>> createQnnLoweringPipeline() {
  std::vector<ir::Pass::ptr_t> ret;
  // Mark IO tensors first, before building the graph
  ret.emplace_back(createQNNGraphIOTensorPass());
  ret.emplace_back(createQNNGraphBuildPass());
  return ret;
}

}  // namespace mllm::qnn