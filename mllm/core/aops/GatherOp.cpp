// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/GatherOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

GatherOp::GatherOp(const GatherOpOptions& options) : BaseOp(OpTypes::kGather), options_(options) {}

void GatherOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::GatherOp>(shared_from_this(), i_irs, o_irs);
}

void GatherOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("GatherOp::forward not implemented in aops base.");
}

void GatherOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& table = inputs[0];
  auto& indices = inputs[1];

  auto shape = table.shape();
  auto indices_shape = indices.shape();
  int dim = options_.dim;
  if (dim < 0) dim += shape.size();

  MLLM_RT_ASSERT(dim >= 0 && dim < shape.size());

  std::vector<int32_t> new_shape;
  new_shape.reserve(dim);
  for (int i = 0; i < dim; ++i) new_shape.push_back(shape[i]);
  for (int s : indices_shape) new_shape.push_back(s);
  for (int i = dim + 1; i < shape.size(); ++i) new_shape.push_back(shape[i]);

  auto o = Tensor::empty(new_shape, table.dtype(), table.device());
  outputs.emplace_back(o);
}

}  // namespace mllm::aops
