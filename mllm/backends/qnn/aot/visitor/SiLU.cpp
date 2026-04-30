// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// SiLU(x) = x * sigmoid(x)
// Decomposed into standard QNN ops: Sigmoid + ElementWiseMultiply

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/SiLU.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTSiLUPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::SiLUOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTSiLUPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto silu_op = op->cast_<mllm::ir::linalg::SiLUOp>();
  if (!silu_op) {
    MLLM_ERROR("Failed to cast to linalg::SiLUOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Input and output tensors
  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Create intermediate tensor for sigmoid output (same shape/dtype as output)
  auto sigmoid_out_tensor = Tensor::empty(o_0->tensor_.shape(), o_0->tensor_.dtype());
  sigmoid_out_tensor.setName(silu_op->getAOp()->getName() + "_sigmoid_out");
  auto sigmoid_out = writer.getContext()->create<ir::tensor::TensorValue>(sigmoid_out_tensor);

  // Copy quantization recipe from output to intermediate if available
  if (op->getAttr("quant_recipe")) {
    sigmoid_out->setAttr("quant_recipe", op->getAttr("quant_recipe"));
  }

  // Step 1: Sigmoid(input) → sigmoid_out
  auto sigmoid_node = QnnAOTNodeOperation::create("Sigmoid");
  sigmoid_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, sigmoid_out))
      ->setName(silu_op->getAOp()->getName() + "_sigmoid");
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, sigmoid_node);

  // Step 2: ElementWiseMultiply(input, sigmoid_out) → output
  auto mul_node = QnnAOTNodeOperation::create("ElementWiseMultiply");
  mul_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, sigmoid_out))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(silu_op->getAOp()->getName());
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, mul_node);

  return true;
}

}  // namespace mllm::qnn::aot
