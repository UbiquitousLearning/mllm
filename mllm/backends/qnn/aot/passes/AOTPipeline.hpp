// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "mllm/compile/passes/Pass.hpp"

namespace mllm::qnn::aot {

std::vector<std::shared_ptr<ir::Pass>> createQnnAOTLoweringPipeline();

}  // namespace mllm::qnn::aot
