// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "mllm/compile/passes/Pass.hpp"

namespace mllm::ir {

struct LLMCanonicalizationPipelineOptions {
  bool auxiliary_dbg_info = true;
};

std::vector<Pass::ptr_t> createLLMCanonicalizationPipeline(const LLMCanonicalizationPipelineOptions& options = {
                                                               .auxiliary_dbg_info = true});
}  // namespace mllm::ir
