// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/visitor/LayerNorm.hpp"

#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/aops/LayerNormOp.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

bool QnnAOTLayerNormPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::LayerNormOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTLayerNormPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto layer_norm_op = op->cast_<mllm::ir::linalg::LayerNormOp>();
  if (!layer_norm_op) {
    MLLM_ERROR("Failed to cast to linalg::LayerNormOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto base_op = layer_norm_op->getAOp();
  auto real_layer_norm_op = dynamic_cast<mllm::aops::LayerNormOp*>(base_op);
  if (!real_layer_norm_op) {
    MLLM_ERROR("Failed to cast BaseOp to mllm::aops::LayerNormOp");
    return false;
  }

  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto qnn_op_node = QnnAOTNodeOperation::create("LayerNorm");
  qnn_op_node->setPackageName("qti.aisw");
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input));

  if (real_layer_norm_op->options().elementwise_affine) {
    auto weight = writer.getContext()
                      ->lookupSymbolTable(base_op->getName() + ".weight")
                      ->outputs()
                      .front()
                      ->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, weight, true));
  }

  if (real_layer_norm_op->options().bias) {
    auto bias = writer.getContext()
                    ->lookupSymbolTable(base_op->getName() + ".bias")
                    ->outputs()
                    .front()
                    ->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, bias, true));
  }

  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("epsilon", real_layer_norm_op->options().eps));

  std::vector<uint32_t> axes_dims = {1};
  auto axes_param = QNNParamTensorWrapper::create("axes", base_op->getName() + "_axes", QNN_DATATYPE_UINT_32, axes_dims);
  auto* axes_data = reinterpret_cast<uint32_t*>(axes_param->alloc());
  axes_data[0] = input->tensor_.shape().size() - 1;
  qnn_op_node->emplaceParamTensor(axes_param);

  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(base_op->getName());

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
