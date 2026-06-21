// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/op/QNNBaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include <vector>

namespace mllm::qnn {

class QNNAddOp final : public aops::AddOp {
 public:
  explicit QNNAddOp(const aops::AddOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QNNAddOpFactory : public TypedOpFactory<OpTypes::kAdd, aops::AddOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::AddOpOptions& options) override {
    return std::make_shared<QNNAddOp>(options);
  }
};

class QNNAddPattern : public QNNBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override { return op->isa_<mllm::ir::linalg::AddOp>(); }

  bool addNode(const std::string& graphName, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;

  static std::pair<OpTypes, std::shared_ptr<QNNAddPattern>> create() {
    return {OpTypes::kAdd, std::make_shared<QNNAddPattern>()};
  }
};

class QNNMulOp final : public aops::MulOp {
 public:
  explicit QNNMulOp(const aops::MulOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QNNMulOpFactory : public TypedOpFactory<OpTypes::kMul, aops::MulOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MulOpOptions& options) override {
    return std::make_shared<QNNMulOp>(options);
  }
};

class QNNMulPattern : public QNNBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override { return op->isa_<mllm::ir::linalg::MulOp>(); }

  bool addNode(const std::string& graphName, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;

  static std::pair<OpTypes, std::shared_ptr<QNNMulPattern>> create() {
    return {OpTypes::kMul, std::make_shared<QNNMulPattern>()};
  }
};

}  // namespace mllm::qnn
