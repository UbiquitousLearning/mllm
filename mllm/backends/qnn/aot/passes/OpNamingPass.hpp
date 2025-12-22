// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::qnn {

class OpNamingPass final : public ir::Pass {
 public:
  OpNamingPass() = default;

  ~OpNamingPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

ir::Pass::ptr_t createOpNamingPass();

}  // namespace mllm::qnn
