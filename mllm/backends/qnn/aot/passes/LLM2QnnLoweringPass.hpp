// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "mllm/core/OpTypes.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/passes/Pass.hpp"
#include "mllm/backends/qnn/aot/visitor/Base.hpp"

namespace mllm::qnn::aot {

class LLM2QnnLoweringPass final : public ir::Pass {
 public:
  explicit LLM2QnnLoweringPass(std::string simple_qnn_graph_name = "");

  ~LLM2QnnLoweringPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

  uint8_t runSimpleGraph(const ir::node_ptr_t& op, const std::string& root_graph_name, const std::string& qnn_graph_name);

 private:
  template<typename... Patterns>
  void registerPatterns() {
    (named_pattern_.insert(Patterns::create()), ...);
  }

  std::unordered_map<OpTypes, std::shared_ptr<QnnAOTBasePattern>> named_pattern_;
  std::unordered_map<std::string, ir::graph::SubGraphOp::ptr_t> subgraph_map_;
  std::string simple_qnn_graph_name_;
};

ir::Pass::ptr_t createLLM2QnnLoweringPass(std::string simple_qnn_graph_name = "");

}  // namespace mllm::qnn::aot
