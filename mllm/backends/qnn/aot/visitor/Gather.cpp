// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/GatherOp.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Gather.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTGatherPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::GatherOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTGatherPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto gather_op = op->cast_<mllm::ir::linalg::GatherOp>();
  if (!gather_op) {
    MLLM_ERROR("Failed to cast to linalg::GatherOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Inputs
  auto table = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto indices = (*std::next(op->inputs().begin()))->cast_<ir::tensor::TensorValue>();

  // Output
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Create QNN Gather Op
  auto qnn_op_node = QnnAOTNodeOperation::create("Gather");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, table))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, indices))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(gather_op->getAOp()->getName());

  // Add scalar param axis
  int axis = dynamic_cast<mllm::aops::GatherOp*>(gather_op->getAOp())->options().dim;
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("axis", (int32_t)axis));

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
