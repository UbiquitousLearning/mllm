// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"

namespace mllm::ir {

class FlattenTensorAndLinalgSymbol2ProgramSymbolPass final : public Pass {
 public:
  FlattenTensorAndLinalgSymbol2ProgramSymbolPass() = default;

  uint8_t run(const node_ptr_t& op) override;
};

Pass::ptr_t createFlattenTensorAndLinalgSymbol2ProgramSymbolPass();

}  // namespace mllm::ir
