// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/TransposeOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

TransposeOp::TransposeOp(const TransposeOpOptions& options) : BaseOp(OpTypes::kTranspose), options_(options) {}

void TransposeOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void TransposeOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::TransposeOp>(shared_from_this(), i_irs, o_irs);
}

void TransposeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("TransposeOp::forward not implemented in aops base.");
}

void TransposeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto input_shape = input.shape();

  // Handle negative indices
  int dim0 = options_.dim0;
  int dim1 = options_.dim1;

  if (dim0 < 0) dim0 += input_shape.size();
  if (dim1 < 0) dim1 += input_shape.size();

  // Validate dimensions
  MLLM_RT_ASSERT(dim0 >= 0 && dim0 < (int)input_shape.size());
  MLLM_RT_ASSERT(dim1 >= 0 && dim1 < (int)input_shape.size());

  // Create output shape by swapping dimensions
  auto output_shape = input_shape;
  std::swap(output_shape[dim0], output_shape[dim1]);

  outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
}

void TransposeOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops