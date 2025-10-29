// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/InterpolateOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

InterpolateOp::InterpolateOp(const InterpolateOpOptions& options) : BaseOp(OpTypes::kInterpolate), options_(options) {}

void InterpolateOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void InterpolateOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::InterpolateOp>(shared_from_this(), i_irs, o_irs);
}

void InterpolateOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("InterpolateOp::forward not implemented in aops base.");
}

void InterpolateOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Get the input tensor
  const auto& input = inputs[0];

  // Skip if input is nil
  if (input.isNil()) { return; }

  // Get input shape
  auto input_shape = input.shape();
  const int input_dim = static_cast<int>(input_shape.size());

  // Calculate output shape based on options
  std::vector<int> output_shape = input_shape;

  // If size is specified, use it to determine output dimensions
  if (!options_.size.empty()) {
    // Ensure size vector has correct dimensions
    MLLM_RT_ASSERT(options_.size.size() <= input_dim);

    // Apply size to the last N dimensions where N is the size of options_.size
    const int offset = input_dim - static_cast<int>(options_.size.size());
    for (size_t i = 0; i < options_.size.size(); ++i) { output_shape[offset + i] = options_.size[i]; }
  }
  // If scale_factor is specified, use it to scale dimensions
  else if (!options_.scale_factor.empty()) {
    // Ensure scale_factor vector has correct dimensions
    MLLM_RT_ASSERT(options_.scale_factor.size() <= input_dim);

    // Apply scale factor to the last N dimensions where N is the size of options_.scale_factor
    const int offset = input_dim - static_cast<int>(options_.scale_factor.size());
    for (size_t i = 0; i < options_.scale_factor.size(); ++i) {
      output_shape[offset + i] = static_cast<int>(input_shape[offset + i] * options_.scale_factor[i]);
    }
  }

  // Create output tensor with the calculated shape
  outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
}

void InterpolateOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
