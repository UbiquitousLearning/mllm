// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/core/aops/RMSNormOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/RMSNorm.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTRMSNormPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::RMSNormOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTRMSNormPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto rms_op = op->cast_<mllm::ir::linalg::RMSNormOp>();
  if (!rms_op) {
    MLLM_ERROR("Failed to cast to linalg::RMSNormOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto a = rms_op->getAOp();
  auto rms_aop = dynamic_cast<mllm::aops::RMSNormOp*>(a);
  if (!rms_aop) {
    MLLM_ERROR("Failed to cast to aops::RMSNormOp");
    return false;
  }

  auto weight =
      writer.getContext()->lookupSymbolTable(a->getName() + ".weight")->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Start to attach
  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();
  auto qnn_op_node = QnnAOTNodeOperation::create("RmsNorm");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceParamScalar(mllm::qnn::QNNParamScalarWrapper::create("epsilon", rms_aop->options().epsilon));

  std::vector<uint32_t> axes_dims = {1};
  auto axes_param = mllm::qnn::QNNParamTensorWrapper::create("axes", a->getName() + "_axes", QNN_DATATYPE_UINT_32, axes_dims);
  uint32_t* axes_data = (uint32_t*)axes_param->alloc();
  axes_data[0] = i_0->tensor_.shape().size() - 1;
  qnn_op_node->emplaceParamTensor(axes_param);

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, weight, true))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(rms_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
