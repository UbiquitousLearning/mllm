// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/op/QNNBaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/aops/CastTypeOp.hpp"
#include "mllm/core/Tensor.hpp"
#include <vector>
#include <memory>

namespace mllm::qnn {

// Forward declarations
class QNNBackend;

class QNNCastTypeOp final : public aops::CastTypeOp {
 public:
  explicit QNNCastTypeOp(const aops::CastTypeOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QNNCastTypeOpFactory : public TypedOpFactory<OpTypes::kCastType, aops::CastTypeOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CastTypeOpOptions& options) override {
    return std::make_shared<QNNCastTypeOp>(options);
  }
};

class QNNCastTypePattern : public QNNBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override { return op->isa_<mllm::ir::linalg::CastTypeOp>(); }

  bool addNode(const std::string& graphName, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;

  static std::pair<OpTypes, std::shared_ptr<QNNCastTypePattern>> create() {
    return {OpTypes::kCastType, std::make_shared<QNNCastTypePattern>()};
  }

 private:
  bool addQuantizeNode(const std::string& graphName, QNNCastTypeOp* qnnCastTypeOp,
                       const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                       const std::vector<ir::tensor::TensorValue::ptr_t>& outputs,
                       const std::shared_ptr<QNNBackend>& qnnBackend);

  bool addDequantizeNode(const std::string& graphName, QNNCastTypeOp* qnnCastTypeOp,
                         const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                         const std::vector<ir::tensor::TensorValue::ptr_t>& outputs,
                         const std::shared_ptr<QNNBackend>& qnnBackend);
};

}  // namespace mllm::qnn
