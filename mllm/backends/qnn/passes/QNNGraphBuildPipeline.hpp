// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp"
#include "mllm/backends/qnn/passes/QNNOpNamingPass.hpp"
#include "mllm/compile/passes/Pass.hpp"

namespace mllm::qnn {

std::vector<std::shared_ptr<ir::Pass>> createQnnLoweringPipeline() {
  std::vector<ir::Pass::ptr_t> ret;
  // Mark IO tensors first, before building the graph
  ret.emplace_back(createQNNGraphIOTensorPass());
  // Assign unique names to unnamed operations
  ret.emplace_back(createQNNOpNamingPass());
  // Build the QNN computation graph
  ret.emplace_back(createQNNGraphBuildPass());
  return ret;
}

}  // namespace mllm::qnn