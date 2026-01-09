// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::qnn::aot {

class LPBQCanonicalizePass final : public ir::Pass {
 public:
  LPBQCanonicalizePass() = default;

  ~LPBQCanonicalizePass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

ir::Pass::ptr_t createLPBQCanonicalizePass();

}  // namespace mllm::qnn::aot
