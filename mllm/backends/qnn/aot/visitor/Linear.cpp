// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Linear.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::qnn::aot {

bool QnnAOTLinearPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::LinearOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTLinearPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto linear_op = op->cast_<mllm::ir::linalg::LinearOp>();
  if (!linear_op) {
    MLLM_ERROR("Failed to cast to linalg::LinearOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto base_op = linear_op->getAOp();
  auto real_linear_op = dynamic_cast<mllm::aops::LinearOp*>(base_op);
  if (!real_linear_op) {
    MLLM_ERROR("Failed to cast BaseOp to mllm::aops::LinearOp");
    return false;
  }

  // Retrieve weight from symbol table
  auto weight_val = writer.getContext()
                        ->lookupSymbolTable(base_op->getName() + ".weight")
                        ->outputs()
                        .front()
                        ->cast_<ir::tensor::TensorValue>();

  // Create QNN FullyConnected Op
  auto qnn_op_node = QnnAOTNodeOperation::create("FullyConnected");
  qnn_op_node->setPackageName("qti.aisw");

  weight_val->tensor_ = weight_val->tensor_.to(kUInt8);

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, weight_val, true));

  // Handle Bias
  if (real_linear_op->options().bias) {
    auto bias_val = writer.getContext()
                        ->lookupSymbolTable(base_op->getName() + ".bias")
                        ->outputs()
                        .front()
                        ->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, bias_val, true));
  }

  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(base_op->getName());

  // Add params: keep_dims
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("keep_dims", false));

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
