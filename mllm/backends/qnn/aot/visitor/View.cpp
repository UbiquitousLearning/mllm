// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/View.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::qnn::aot {

bool QnnAOTViewPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::ViewOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTViewPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto view_op = op->cast_<mllm::ir::linalg::ViewOp>();
  if (!view_op) {
    MLLM_ERROR("Failed to cast to linalg::ViewOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Input
  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();

  // Output
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Create Shape Tensor
  auto output_shape = output->tensor_.shape();
  std::vector<int32_t> shape_data;
  for (auto dim : output_shape) { shape_data.push_back(static_cast<int32_t>(dim)); }

  // Shape tensor shape is [rank(output)]
  std::vector<int32_t> shape_tensor_shape = {static_cast<int32_t>(shape_data.size())};

  // Create QNN Reshape Op
  auto qnn_op_node = QnnAOTNodeOperation::create("Reshape");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(view_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
