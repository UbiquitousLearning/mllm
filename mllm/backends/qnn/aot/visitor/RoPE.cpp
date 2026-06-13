// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Lowers RoPE (Rotary Position Embedding) to the custom HTP op from LLaMAPackage.
// The custom op signature: RoPE(input, sin, cos, h_cnt; pose_type) → output
// It supports partial rotation natively via the HVX kernel.

#include "mllm/utils/Common.hpp"
#include "mllm/core/aops/RoPEOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/RoPE.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTRoPEPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::RoPEOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTRoPEPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto rope_op = op->cast_<mllm::ir::linalg::RoPEOp>();
  if (!rope_op) {
    MLLM_ERROR("Failed to cast to linalg::RoPEOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto a = rope_op->getAOp();
  auto rope_aop = dynamic_cast<mllm::aops::RoPEOp*>(a);
  if (!rope_aop) {
    MLLM_ERROR("Failed to cast to aops::RoPEOp");
    return false;
  }

  // RoPE inputs: x, sin, cos
  auto inputs_it = op->inputs().begin();
  auto i_0 = (*inputs_it)->cast_<ir::tensor::TensorValue>();          // input tensor
  auto i_sin = (*std::next(inputs_it))->cast_<ir::tensor::TensorValue>();  // sin embeddings
  auto i_cos = (*std::next(inputs_it, 2))->cast_<ir::tensor::TensorValue>();  // cos embeddings

  // RoPE output
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Create the custom HTP RoPE op from LLaMAPackage
  auto qnn_op_node = QnnAOTNodeOperation::create("RoPE");
  qnn_op_node->setPackageName("LLaMAPackage");

  // pose_type parameter: 0 for standard RoPE
  // The custom HTP op uses this to select between different RoPE variants
  qnn_op_node->emplaceParamScalar(mllm::qnn::QNNParamScalarWrapper::create("pose_type", static_cast<uint32_t>(0)));

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_sin))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_cos))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(rope_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
