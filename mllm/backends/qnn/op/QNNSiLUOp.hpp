// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/op/QNNBaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/aops/SiLUOp.hpp"
#include <vector>

namespace mllm::qnn {

class QNNSiLUOp final : public aops::SiLUOp {
 public:
  explicit QNNSiLUOp(const aops::SiLUOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QNNSiLUOpFactory : public TypedOpFactory<OpTypes::kSiLU, aops::SiLUOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SiLUOpOptions& options) override {
    return std::make_shared<QNNSiLUOp>(options);
  }
};

class QNNSiLUPattern : public QNNBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override { return op->isa_<mllm::ir::linalg::SiLUOp>(); }

  bool addNode(const std::string& graphName, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;

  static std::pair<OpTypes, std::shared_ptr<QNNSiLUPattern>> create() {
    return {OpTypes::kSiLU, std::make_shared<QNNSiLUPattern>()};
  }
};

}  // namespace mllm::qnn
