// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Transpose.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/TransposeOp.hpp"
#include <cstring>

namespace mllm::qnn::aot {

bool QnnAOTTransposePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::TransposeOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTTransposePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto transpose_op = op->cast_<mllm::ir::linalg::TransposeOp>();
  if (!transpose_op) {
    MLLM_ERROR("Failed to cast to linalg::TransposeOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto base_op = transpose_op->getAOp();
  auto real_transpose_op = dynamic_cast<mllm::aops::TransposeOp*>(base_op);
  if (!real_transpose_op) {
    MLLM_ERROR("Failed to cast BaseOp to mllm::aops::TransposeOp");
    return false;
  }

  const auto& options = real_transpose_op->options();

  // Calculate perm
  auto input_shape = input->tensor_.shape();
  int rank = input_shape.size();

  std::vector<uint32_t> perm(rank);
  for (int i = 0; i < rank; ++i) { perm[i] = i; }

  if (options.dim0 < rank && options.dim1 < rank) { std::swap(perm[options.dim0], perm[options.dim1]); }

  // Create QNN Op Node
  auto qnn_op_node = QnnAOTNodeOperation::create("Transpose");
  qnn_op_node->setPackageName("qti.aisw");

  // Add Input
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input));

  // Add Perm Param
  auto permName = base_op->getName() + ".perm";
  auto permParam = QNNParamTensorWrapper::create("perm", permName, QNN_DATATYPE_UINT_32, std::vector<uint32_t>{(uint32_t)rank});
  uint32_t* permData = static_cast<uint32_t*>(permParam->alloc());
  std::memcpy(permData, perm.data(), rank * sizeof(uint32_t));
  qnn_op_node->emplaceParamTensor(permParam);

  // Add Output
  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(base_op->getName());

  // Register
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
