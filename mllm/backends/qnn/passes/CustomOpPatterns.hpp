// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/op/QNNBaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include <vector>

namespace mllm::qnn {

class QNNDequantizeAddPattern : public QNNBasePattern {
 public:
  bool isMatch(const ir::op_ptr_t& op) override { return op->isa_<ir::linalg::CustomizedOp>(); }

  bool addNode(const std::string& graphName, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;
};

}  // namespace mllm::qnn