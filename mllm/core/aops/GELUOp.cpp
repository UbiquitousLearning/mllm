// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/GELUOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

GELUOp::GELUOp(const GELUOpOptions& options) : BaseOp(OpTypes::kGELU), options_(options) {}

void GELUOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void GELUOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::GELUOp>(shared_from_this(), i_irs, o_irs);
}

void GELUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("GELUOp::forward not implemented in aops base.");
}

void GELUOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void GELUOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops