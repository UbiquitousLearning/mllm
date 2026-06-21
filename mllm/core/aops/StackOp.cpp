// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/StackOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

StackOp::StackOp(const StackOpOptions& options) : BaseOp(OpTypes::kStack), options_(options) {}

void StackOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void StackOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::StackOp>(shared_from_this(), i_irs, o_irs);
}

void StackOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("StackOp::forward not implemented in aops base.");
}

void StackOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (inputs.empty()) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "StackOp: no inputs");
    return;
  }

  const int n_dims = inputs[0].shape().size();
  int at_dim = options_.dim;

  // Normalize dim into [0, n_dims]
  if (at_dim < 0) { at_dim += (n_dims + 1); }
  if (at_dim < 0 || at_dim > n_dims) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "StackOp: dim {} out of range [0, {}]", at_dim, n_dims);
    return;
  }

  // Check all input shapes equal
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].shape() != inputs[0].shape()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "StackOp: input shape mismatch");
      return;
    }
  }

  // Build new shape by inserting a new dimension of size inputs.size()
  std::vector<int> new_shape;
  new_shape.reserve(n_dims + 1);
  for (int d = 0; d < at_dim; ++d) { new_shape.push_back(inputs[0].shape()[d]); }
  new_shape.push_back(static_cast<int>(inputs.size()));
  for (int d = at_dim; d < n_dims; ++d) { new_shape.push_back(inputs[0].shape()[d]); }

  outputs.emplace_back(Tensor::empty(new_shape, inputs[0].dtype(), inputs[0].device()));
}

void StackOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops