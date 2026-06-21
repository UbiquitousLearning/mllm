// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/RepeatOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

RepeatOp::RepeatOp(const RepeatOpOptions& options) : BaseOp(OpTypes::kRepeat), options_(options) {}

void RepeatOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void RepeatOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::RepeatOp>(shared_from_this(), i_irs, o_irs);
}

void RepeatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("RepeatOp::forward not implemented in aops base.");
}

void RepeatOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto i_shape = i.shape();
  i_shape[options_.dim] *= options_.repeat_times;
  outputs.emplace_back(Tensor::empty(i_shape, i.dtype(), i.device()));
}

void RepeatOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops