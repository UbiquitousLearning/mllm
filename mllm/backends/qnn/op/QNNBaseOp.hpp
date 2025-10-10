// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pattern.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include <vector>

namespace mllm::qnn {

class QNNBasePattern : public ir::Pattern {
 public:
  QNNBasePattern() = default;

  bool isMatch(const mllm::ir::op_ptr_t& op) override { return false; }

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override { return false; }

  virtual bool addNode(const std::string& graphName, const ir::op_ptr_t& op,
                       const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                       const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) = 0;
};
}  // namespace mllm::qnn
