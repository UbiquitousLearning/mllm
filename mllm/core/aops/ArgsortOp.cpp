// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ArgsortOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

ArgsortOp::ArgsortOp(const ArgsortOpOptions& options) : BaseOp(OpTypes::kArgsort), options_(options) {}

void ArgsortOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ArgsortOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ArgsortOp>(shared_from_this(), i_irs, o_irs);
}

void ArgsortOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("ArgsortOp::forward not implemented in aops base.");
}

void ArgsortOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Define output tensor shapes based on input shapes
  auto& input = inputs[0];

  if (!input.isNil()) {
    auto input_shape = input.shape();
    // Output indices tensor has the same shape as input
    outputs.emplace_back(Tensor::empty(input_shape, kInt32, input.device()));
  }
}

void ArgsortOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
