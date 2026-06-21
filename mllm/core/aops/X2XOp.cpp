// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/X2XOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

X2XOp::X2XOp(const X2XOpOptions& options) : BaseOp(OpTypes::kX2X), options_(options) {}

void X2XOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void X2XOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::X2XOp>(shared_from_this(), i_irs, o_irs);
}

void X2XOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("X2XOp::forward not implemented in aops base.");
}

void X2XOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), options_.device));
}

void X2XOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops