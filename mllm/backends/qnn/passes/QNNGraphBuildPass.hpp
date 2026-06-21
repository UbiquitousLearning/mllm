// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/backends/qnn/op/QNNBaseOp.hpp"

namespace mllm::qnn {

class QNNGraphBuildPass final : public ir::Pass {
 public:
  QNNGraphBuildPass();

  ~QNNGraphBuildPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

 private:
  template<typename T>
  void _reg_one_pattern() {
    auto pair = T::create();
    patterns_.insert({pair.first, pair.second});
  }

  template<typename... Args>
  void regPattern() {
    (_reg_one_pattern<Args>(), ...);
  }

  void buildQnnGraph(const ir::graph::SubGraphOp::ptr_t& sub_graph_op);

  std::unordered_map<OpTypes, std::shared_ptr<QNNBasePattern>> patterns_;
};

ir::Pass::ptr_t createQNNGraphBuildPass() { return std::make_shared<QNNGraphBuildPass>(); }

}  // namespace mllm::qnn