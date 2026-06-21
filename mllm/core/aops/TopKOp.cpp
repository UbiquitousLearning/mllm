// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/TopKOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

TopKOp::TopKOp(const TopKOpOptions& options) : BaseOp(OpTypes::kTopK), options_(options) {}

void TopKOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void TopKOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::TopKOp>(shared_from_this(), i_irs, o_irs);
}

void TopKOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("TopKOp::forward not implemented in aops base.");
}

void TopKOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Define output tensor shapes based on input shapes
  auto& input = inputs[0];

  if (!input.isNil()) {
    auto input_shape = input.shape();
    std::vector<int32_t> values_shape(input_shape);
    std::vector<int32_t> indices_shape(input_shape);

    int32_t dim = options_.dim;
    // Handle negative dimension index
    if (dim < 0) { dim += input_shape.size(); }

    // Replace the specified dimension with k
    values_shape[dim] = options_.k;
    indices_shape[dim] = options_.k;

    // Create two output tensors: values and indices
    outputs.emplace_back(Tensor::empty(values_shape, input.dtype(), input.device()));
    outputs.emplace_back(Tensor::empty(indices_shape, kInt32, input.device()));
  }
}

void TopKOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
