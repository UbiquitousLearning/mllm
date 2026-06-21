// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "mllm/core/OpTypes.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/passes/Pass.hpp"
#include "mllm/backends/qnn/aot/visitor/Base.hpp"

namespace mllm::qnn::aot {

class LLM2QnnLoweringPass final : public ir::Pass {
 public:
  LLM2QnnLoweringPass();

  ~LLM2QnnLoweringPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

 private:
  template<typename... Patterns>
  void registerPatterns() {
    (named_pattern_.insert(Patterns::create()), ...);
  }

  std::unordered_map<OpTypes, std::shared_ptr<QnnAOTBasePattern>> named_pattern_;
  std::unordered_map<std::string, ir::graph::SubGraphOp::ptr_t> subgraph_map_;
};

ir::Pass::ptr_t createLLM2QnnLoweringPass();

}  // namespace mllm::qnn::aot
