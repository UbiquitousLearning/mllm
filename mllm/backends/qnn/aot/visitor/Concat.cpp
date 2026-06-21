// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Concat.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/ConcatOp.hpp"

namespace mllm::qnn::aot {

bool QnnAOTConcatPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::ConcatOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTConcatPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto concat_op = op->cast_<mllm::ir::linalg::ConcatOp>();
  if (!concat_op) {
    MLLM_ERROR("Failed to cast to linalg::ConcatOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto base_op = concat_op->getAOp();
  auto real_concat_op = dynamic_cast<mllm::aops::ConcatOp*>(base_op);
  if (!real_concat_op) {
    MLLM_ERROR("Failed to cast BaseOp to mllm::aops::ConcatOp");
    return false;
  }

  int axis = real_concat_op->options().dim;

  // Handle negative axis
  // We can use the first input to determine rank, assuming all inputs have same rank
  auto first_input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto input_shape = first_input->tensor_.shape();
  int rank = input_shape.size();
  if (axis < 0) { axis += rank; }

  // Create QNN Op Node
  auto qnn_op_node = QnnAOTNodeOperation::create("Concat");
  qnn_op_node->setPackageName("qti.aisw");

  // Add Inputs
  for (auto& input_val : op->inputs()) {
    auto input = input_val->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input));
  }

  // Add Params
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("axis", (uint32_t)axis));

  // Add Output
  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(base_op->getName());

  // Register
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
