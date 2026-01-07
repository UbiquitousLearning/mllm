// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Repeat.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/RepeatOp.hpp"
#include <cstring>

namespace mllm::qnn::aot {

bool QnnAOTRepeatPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::RepeatOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTRepeatPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto repeat_op = op->cast_<mllm::ir::linalg::RepeatOp>();
  if (!repeat_op) {
    MLLM_ERROR("Failed to cast to linalg::RepeatOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto base_op = repeat_op->getAOp();
  auto real_repeat_op = dynamic_cast<mllm::aops::RepeatOp*>(base_op);
  if (!real_repeat_op) {
    MLLM_ERROR("Failed to cast BaseOp to mllm::aops::RepeatOp");
    return false;
  }

  const auto& options = real_repeat_op->options();
  int dim = options.dim;
  int repeat_times = options.repeat_times;

  auto input_shape = input->tensor_.shape();
  int rank = input_shape.size();

  if (dim < 0) { dim += rank; }

  std::vector<uint32_t> multiples(rank, 1);
  if (dim >= 0 && dim < rank) {
    multiples[dim] = (uint32_t)repeat_times;
  } else {
    MLLM_ERROR("Invalid dimension for RepeatOp: {}", dim);
    return false;
  }

  // Create QNN Op Node
  // QNN uses "Tile" for repeat
  auto qnn_op_node = QnnAOTNodeOperation::create("Tile");
  qnn_op_node->setPackageName("qti.aisw");

  // Add Input
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input));

  // Add multiples Param
  auto multiplesName = base_op->getName() + ".multiples";
  auto multiplesParam =
      QNNParamTensorWrapper::create("multiples", multiplesName, QNN_DATATYPE_UINT_32, std::vector<uint32_t>{(uint32_t)rank});
  uint32_t* multiplesData = static_cast<uint32_t*>(multiplesParam->alloc());
  std::memcpy(multiplesData, multiples.data(), rank * sizeof(uint32_t));
  qnn_op_node->emplaceParamTensor(multiplesParam);

  // Add Output
  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(base_op->getName());

  // Register
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
