// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/op/QNNBaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/aops/TransposeOp.hpp"
#include <vector>

namespace mllm::qnn {

class QNNTransposeOp final : public aops::TransposeOp {
 public:
  explicit QNNTransposeOp(const aops::TransposeOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QNNTransposeOpFactory : public TypedOpFactory<OpTypes::kTranspose, aops::TransposeOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::TransposeOpOptions& options) override {
    return std::make_shared<QNNTransposeOp>(options);
  }
};

class QNNTransposePattern : public QNNBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override { return op->isa_<mllm::ir::linalg::TransposeOp>(); }

  bool addNode(const std::string& graphName, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;

  static std::pair<OpTypes, std::shared_ptr<QNNTransposePattern>> create() {
    return {OpTypes::kTranspose, std::make_shared<QNNTransposePattern>()};
  }
};

}  // namespace mllm::qnn
