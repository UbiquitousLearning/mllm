// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"

namespace mllm::ir {

class ExtractSymbolsPass final : public Pass {
 public:
  ExtractSymbolsPass() = default;

  uint8_t run(const node_ptr_t& op) override;
};

Pass::ptr_t createExtractSymbolsPass();

}  // namespace mllm::ir
