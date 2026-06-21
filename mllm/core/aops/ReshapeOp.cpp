// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ReshapeOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

ReshapeOp::ReshapeOp(const ReshapeOpOptions& options) : BaseOp(OpTypes::kReshape), options_(options) {}

void ReshapeOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ReshapeOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ReshapeOp>(shared_from_this(), i_irs, o_irs);
}

void ReshapeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("ReshapeOp::forward not implemented in aops base.");
}

void ReshapeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& it = inputs[0];
  auto const& new_shape = options_.shape;

  std::vector<int32_t> actual_shape = new_shape;
  int infer_dim = -1;
  size_t product = 1;

  for (size_t i = 0; i < actual_shape.size(); ++i) {
    if (actual_shape[i] == -1) {
      // only one dimension can be inferred
      MLLM_RT_ASSERT(infer_dim == -1);
      infer_dim = static_cast<int>(i);
    } else {
      product *= actual_shape[i];
    }
  }

  // infer dim
  if (infer_dim != -1) {
    size_t input_numel = it.numel();
    MLLM_RT_ASSERT(product != 0);
    MLLM_RT_ASSERT(input_numel % product == 0);
    actual_shape[infer_dim] = static_cast<int32_t>(input_numel / product);
  }

  // check numel
  size_t new_numel = 1;
  for (int dim : actual_shape) { new_numel *= dim; }
  MLLM_RT_ASSERT_EQ(it.numel(), new_numel);

  outputs.emplace_back(Tensor::empty(actual_shape, it.dtype(), it.device()));
}

void ReshapeOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops