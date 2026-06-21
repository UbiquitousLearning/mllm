// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/aops/ReduceOps.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Reduce.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include <cstring>

namespace mllm::qnn::aot {

template<typename IROpType, typename AOpType>
static bool rewriteReduceOp(ir::IRWriter& writer, const ir::op_ptr_t& op, const std::string& qnnOpName) {
  auto env = AOTCompileContext::getInstance().getEnv();
  auto reduce_op = op->cast_<IROpType>();
  if (!reduce_op) {
    MLLM_ERROR("Failed to cast to ReduceOp for {}", qnnOpName);
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto base_op = reduce_op->getAOp();
  auto aop = dynamic_cast<AOpType*>(base_op);
  if (!aop) {
    MLLM_ERROR("Failed to cast base op to specific ReduceOp");
    return false;
  }

  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  int32_t dim = aop->options().dim;
  bool keep_dims = aop->options().keep_dim;

  auto input_rank = i_0->tensor_.rank();
  if (dim < 0) { dim += static_cast<int32_t>(input_rank); }

  // Create QNN Op Node
  auto qnn_op_node = QnnAOTNodeOperation::create(qnnOpName);
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(base_op->getName());

  // axes
  auto axesName = base_op->getName() + ".axes";
  auto axesParam = mllm::qnn::QNNParamTensorWrapper::create("axes", axesName, QNN_DATATYPE_UINT_32, std::vector<uint32_t>{1});
  uint32_t* axesData = static_cast<uint32_t*>(axesParam->alloc());
  axesData[0] = static_cast<uint32_t>(dim);
  qnn_op_node->emplaceParamTensor(axesParam);

  // keep_dims
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("keep_dims", keep_dims));

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);
  return true;
}

bool QnnAOTReduceMaxPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::ReduceMaxOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTReduceMaxPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  return rewriteReduceOp<mllm::ir::linalg::ReduceMaxOp, mllm::aops::ReduceMaxOp>(writer, op, "ReduceMax");
}

bool QnnAOTReduceMinPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::ReduceMinOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTReduceMinPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  return rewriteReduceOp<mllm::ir::linalg::ReduceMinOp, mllm::aops::ReduceMinOp>(writer, op, "ReduceMin");
}

bool QnnAOTReduceMeanPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::MeanOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTReduceMeanPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  return rewriteReduceOp<mllm::ir::linalg::MeanOp, mllm::aops::MeanOp>(writer, op, "ReduceMean");
}

bool QnnAOTReduceSumPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::ReduceSumOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTReduceSumPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  return rewriteReduceOp<mllm::ir::linalg::ReduceSumOp, mllm::aops::ReduceSumOp>(writer, op, "ReduceSum");
}

}  // namespace mllm::qnn::aot
