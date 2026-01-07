// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Index.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/IndexOp.hpp"

namespace mllm::qnn::aot {

bool QnnAOTIndexPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::IndexOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTIndexPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto index_op = op->cast_<mllm::ir::linalg::IndexOp>();
  if (!index_op) {
    MLLM_ERROR("Failed to cast to linalg::IndexOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto base_op = index_op->getAOp();
  auto real_index_op = dynamic_cast<mllm::aops::IndexOp*>(base_op);
  if (!real_index_op) {
    MLLM_ERROR("Failed to cast BaseOp to mllm::aops::IndexOp");
    return false;
  }

  const auto& indices_list = real_index_op->options().indices_;
  int axis = 0;
  mllm::Tensor indices_tensor;
  bool found_axis = false;

  for (const auto& blob : indices_list) {
    if (blob.slice_indices_.has_value()) {
      auto slice = blob.slice_indices_.value();
      if (slice.start_ == mllm::kAll && slice.end_ == mllm::kAll) {
        if (!found_axis) axis++;
        continue;
      }
    }

    if (blob.vector_indices_.has_value()) {
      if (found_axis) {
        MLLM_ERROR("QNN Gather only supports one axis");
        return false;
      }
      auto vec = blob.vector_indices_.value();
      const std::vector<int32_t>& indices_data = vec;
      indices_tensor = mllm::Tensor::fromVector(indices_data, {(int)indices_data.size()}, mllm::kInt32, mllm::kCPU);
      found_axis = true;
    } else if (blob.tensor_indices_.has_value()) {
      if (found_axis) {
        MLLM_ERROR("QNN Gather only supports one axis");
        return false;
      }
      indices_tensor = blob.tensor_indices_.value();
      found_axis = true;
    }

    if (found_axis) break;
    axis++;
  }

  if (!found_axis) {
    MLLM_ERROR("No indices found for Gather in IndexOp");
    return false;
  }

  indices_tensor.setName(base_op->getName() + ".indices");
  auto indices_tv = mllm::ir::tensor::TensorValue::build(writer.getContext().get(), indices_tensor);

  auto qnn_op_node = QnnAOTNodeOperation::create("Gather");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, indices_tv))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->emplaceParamScalar(QNNParamScalarWrapper::create("axis", (int32_t)axis))
      ->setName(base_op->getName());

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
