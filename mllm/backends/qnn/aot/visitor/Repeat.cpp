// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Repeat.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/RepeatOp.hpp"
#include "mllm/core/Tensor.hpp"
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

  std::vector<uint32_t> multiples(rank + 1, 1);
  if (dim >= 0 && dim < rank) {
    multiples[dim + 1] = (uint32_t)repeat_times;
  } else {
    MLLM_ERROR("Invalid dimension for RepeatOp: {}", dim);
    return false;
  }

  // mllm Repeat semantics are repeat_interleave along one dimension. QNN Tile
  // repeats whole dimensions, so use reshape + tile + reshape:
  // [.., D, ..] -> [.., D, 1, ..] -> tile inserted dim -> [.., D * repeat, ..].
  auto expanded_shape = input_shape;
  expanded_shape.insert(expanded_shape.begin() + dim + 1, 1);
  auto tiled_shape = expanded_shape;
  tiled_shape[dim + 1] = repeat_times;

  auto expanded = writer.getContext()->create<ir::tensor::TensorValue>(
      Tensor::empty(expanded_shape, input->tensor_.dtype(), input->tensor_.device()));
  expanded->setAttr("quant_recipe", input->getAttr("quant_recipe"));
  auto tiled = writer.getContext()->create<ir::tensor::TensorValue>(
      Tensor::empty(tiled_shape, input->tensor_.dtype(), input->tensor_.device()));
  tiled->setAttr("quant_recipe", input->getAttr("quant_recipe"));

  auto reshape_in = QnnAOTNodeOperation::create("Reshape");
  reshape_in->setPackageName("qti.aisw");
  reshape_in->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, expanded))
      ->setName(base_op->getName() + ".repeat_interleave_reshape_in");
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, reshape_in);

  auto tile = QnnAOTNodeOperation::create("Tile");
  tile->setPackageName("qti.aisw");
  tile->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, expanded));
  auto multiplesName = base_op->getName() + ".multiples";
  auto multiplesParam = QNNParamTensorWrapper::create("multiples", multiplesName, QNN_DATATYPE_UINT_32,
                                                     std::vector<uint32_t>{(uint32_t)multiples.size()});
  uint32_t* multiplesData = static_cast<uint32_t*>(multiplesParam->alloc());
  std::memcpy(multiplesData, multiples.data(), multiples.size() * sizeof(uint32_t));
  tile->emplaceParamTensor(multiplesParam);
  tile->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, tiled))
      ->setName(base_op->getName() + ".repeat_interleave_tile");
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, tile);

  auto reshape_out = QnnAOTNodeOperation::create("Reshape");
  reshape_out->setPackageName("qti.aisw");
  reshape_out->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, tiled))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(base_op->getName());
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, reshape_out);

  return true;
}

}  // namespace mllm::qnn::aot
