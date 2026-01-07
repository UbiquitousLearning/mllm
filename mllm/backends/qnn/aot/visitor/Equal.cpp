// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Equal.hpp"
#include <iterator>
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTEqualPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::EqualOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTEqualPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto equal_op = op->cast_<mllm::ir::linalg::EqualOp>();
  if (!equal_op) {
    MLLM_ERROR("Failed to cast to linalg::EqualOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Inputs
  auto input0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto input1 = (*std::next(op->inputs().begin()))->cast_<ir::tensor::TensorValue>();

  // Output
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Create QNN ElementWiseEqual Op
  auto qnn_op_node = QnnAOTNodeOperation::create("ElementWiseEqual");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input0))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input1))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(equal_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
