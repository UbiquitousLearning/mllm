// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Embedding.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTEmbeddingPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::EmbeddingOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTEmbeddingPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto embedding_op = op->cast_<mllm::ir::linalg::EmbeddingOp>();
  if (!embedding_op) {
    MLLM_ERROR("Failed to cast to linalg::EmbeddingOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto a_op = embedding_op->getAOp();

  // Retrieve weight from symbol table
  auto weight =
      writer.getContext()->lookupSymbolTable(a_op->getName() + ".weight")->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Inputs: Indices
  auto indices = op->inputs().front()->cast_<ir::tensor::TensorValue>();

  // Output
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Create QNN Gather Op
  auto qnn_op_node = QnnAOTNodeOperation::create("Gather");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, weight, true))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, indices))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(embedding_op->getAOp()->getName());

  // Add scalar param axis = 0
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("axis", (int32_t)0));

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
