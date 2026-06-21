// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/op/QNNBaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/aops/RMSNormOp.hpp"
#include <vector>

namespace mllm::qnn {

class QNNRMSNormOp final : public aops::RMSNormOp {
 public:
  explicit QNNRMSNormOp(const aops::RMSNormOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QNNRMSNormOpFactory : public TypedOpFactory<OpTypes::kRMSNorm, aops::RMSNormOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RMSNormOpOptions& options) override {
    return std::make_shared<QNNRMSNormOp>(options);
  }
};

class QNNRMSNormPattern : public QNNBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override { return op->isa_<mllm::ir::linalg::RMSNormOp>(); }

  bool addNode(const std::string& graphName, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;

  static std::pair<OpTypes, std::shared_ptr<QNNRMSNormPattern>> create() {
    return {OpTypes::kRMSNorm, std::make_shared<QNNRMSNormPattern>()};
  }
};

}  // namespace mllm::qnn
