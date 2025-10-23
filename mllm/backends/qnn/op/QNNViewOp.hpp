// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/op/QNNBaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include <vector>

namespace mllm::qnn {

class QNNViewOp final : public aops::ViewOp {
 public:
  explicit QNNViewOp(const aops::ViewOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QNNViewOpFactory : public TypedOpFactory<OpTypes::kView, aops::ViewOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ViewOpOptions& options) override {
    return std::make_shared<QNNViewOp>(options);
  }
};

class QNNViewPattern : public QNNBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override { return op->isa_<mllm::ir::linalg::ViewOp>(); }

  bool addNode(const std::string& graphName, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;

  static std::pair<OpTypes, std::shared_ptr<QNNViewPattern>> create() {
    return {OpTypes::kView, std::make_shared<QNNViewPattern>()};
  }
};

}  // namespace mllm::qnn
