// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Softmax.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/SoftmaxOp.hpp"

namespace mllm::qnn::aot {

bool QnnAOTSoftmaxPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::SoftmaxOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTSoftmaxPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto softmax_op = op->cast_<mllm::ir::linalg::SoftmaxOp>();
  if (!softmax_op) {
    MLLM_ERROR("Failed to cast to linalg::SoftmaxOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto base_op = softmax_op->getAOp();
  auto real_softmax_op = dynamic_cast<mllm::aops::SoftmaxOp*>(base_op);
  if (!real_softmax_op) {
    MLLM_ERROR("Failed to cast BaseOp to mllm::aops::SoftmaxOp");
    return false;
  }

  int axis = real_softmax_op->options().axis;
  float beta = 1.0f;  // Default beta

  // Handle negative axis
  auto input_shape = input->tensor_.shape();
  int rank = input_shape.size();
  if (axis < 0) { axis += rank; }

  // Create QNN Op Node
  auto qnn_op_node = QnnAOTNodeOperation::create("Softmax");
  qnn_op_node->setPackageName("qti.aisw");

  // Add Input
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input));

  // Add Params
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("axis", (uint32_t)axis));
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("beta", beta));

  // Add Output
  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(base_op->getName());

  // Register
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
