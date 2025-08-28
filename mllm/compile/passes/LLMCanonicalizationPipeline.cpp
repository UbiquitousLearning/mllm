// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/passes/LLMCanonicalizationPipeline.hpp"
#include "mllm/compile/passes/ExtractSymbolsPass.hpp"
#include "mllm/compile/passes/EliminateDbgInfoPass.hpp"

namespace mllm::ir {

std::vector<Pass::ptr_t> createLLMCanonicalizationPipeline(const LLMCanonicalizationPipelineOptions& options) {
  std::vector<Pass::ptr_t> ret;

  ret.push_back(createExtractSymbolsPass());

  if (!options.auxiliary_dbg_info) { ret.push_back(createEliminateDbgInfoPass()); }
  return ret;
}

}  // namespace mllm::ir
