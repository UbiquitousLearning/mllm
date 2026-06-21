// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/QuickGELUOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

QuickGELUOp::QuickGELUOp(const QuickGELUOpOptions& options) : BaseOp(OpTypes::kQuickGELU), options_(options) {}

void QuickGELUOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void QuickGELUOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::QuickGELUOp>(shared_from_this(), i_irs, o_irs);
}

void QuickGELUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("QuickGELUOp::forward not implemented in aops base.");
}

void QuickGELUOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void QuickGELUOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops