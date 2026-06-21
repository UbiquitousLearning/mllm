// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include <vector>

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/core/aops/SliceOp.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Slice.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTSlicePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::SliceOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTSlicePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto slice_op = op->cast_<mllm::ir::linalg::SliceOp>();
  if (!slice_op) {
    MLLM_ERROR("Failed to cast to linalg::SliceOp");
    return false;
  }

  auto aop = dynamic_cast<mllm::aops::SliceOp*>(slice_op->getAOp());
  if (!aop) {
    MLLM_ERROR("Failed to cast AOp to aops::SliceOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Inputs
  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  // Outputs
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Calc params
  const auto& indices = aop->options().indices_;
  size_t rank = indices.size();

  std::vector<int32_t> ranges_data;
  uint32_t begin_mask = 0;
  uint32_t end_mask = 0;

  for (size_t i = 0; i < rank; ++i) {
    int32_t start = indices[i].start_;
    int32_t end = indices[i].end_;
    int32_t step = indices[i].step_;

    if (start == mllm::kAll) {
      start = 0;
      begin_mask |= (1 << i);
    }
    if (end == mllm::kAll) {
      end = 0;
      end_mask |= (1 << i);
    }

    // Handle [-1] case where start=-1, end=0 (from SliceIndicesPair(-1))
    // If generated from single index v=-1, end is v+1 = 0.
    if (start < 0 && end == 0) { end_mask |= (1 << i); }

    ranges_data.push_back(start);
    ranges_data.push_back(end);
    ranges_data.push_back(step);
  }

  // Create QNN StridedSlice Op
  auto qnn_op_node = QnnAOTNodeOperation::create("StridedSlice");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(slice_op->getAOp()->getName());

  // Params
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create<uint32_t>("begin_mask", begin_mask));
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create<uint32_t>("end_mask", end_mask));
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create<uint32_t>("shrink_axes", 0));
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create<uint32_t>("new_axes_mask", 0));

  // ranges
  auto ranges_param = QNNParamTensorWrapper::create("ranges", slice_op->getAOp()->getName() + ".ranges", QNN_DATATYPE_INT_32,
                                                    std::vector<uint32_t>{(uint32_t)rank, 3});
  int32_t* ptr = static_cast<int32_t*>(ranges_param->alloc());
  if (ptr) { std::memcpy(ptr, ranges_data.data(), ranges_data.size() * sizeof(int32_t)); }
  qnn_op_node->emplaceParamTensor(ranges_param);

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
