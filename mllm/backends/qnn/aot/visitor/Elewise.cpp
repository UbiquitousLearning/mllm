// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/visitor/Elewise.hpp"
#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::qnn::aot {

bool QnnAOTAddPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::AddOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTAddPattern::addNode(const std::string& g_name, const ir::op_ptr_t& op,
                               const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  auto add_op = op->cast_<mllm::ir::linalg::AddOp>();
  if (!add_op) {
    MLLM_ERROR("Failed to cast to linalg::AddOp");
    return false;
  }

  auto* mllm_op = dynamic_cast<aops::AddOp*>(add_op->getAOp());
  if (!mllm_op) {
    MLLM_ERROR("Failed to cast to aops::AddOp");
    return false;
  }

  // TODO

  return true;
}

bool QnnAOTAddQuantRecipePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::AddOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTAddQuantRecipePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // TODO
  return true;
}

}  // namespace mllm::qnn::aot
