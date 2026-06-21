// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Sigmoid.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTSigmoidPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::SigmoidOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTSigmoidPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto sigmoid_op = op->cast_<mllm::ir::linalg::SigmoidOp>();
  if (!sigmoid_op) {
    MLLM_ERROR("Failed to cast to linalg::SigmoidOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Inputs
  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();

  // Outputs
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Create QNN Sigmoid Op
  auto qnn_op_node = QnnAOTNodeOperation::create("Sigmoid");
  // Sigmoid is likely a basic op, so maybe no package name needed, or "qti.aisw"?
  // Following previous pattern for elementwise ops (Add/Mul/Neg/Equal where I checked one),
  // basic elementwise ops work without package name usually, but let's check what I did for Equal.
  // In Equal I commented it out. In Add/Mul it wasn't there.
  // Let's stick to not setting it for basic elementwise unless necessary.

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(sigmoid_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
