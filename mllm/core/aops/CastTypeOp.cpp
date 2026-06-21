// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/CastTypeOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

CastTypeOp::CastTypeOp(const CastTypeOpOptions& options) : BaseOp(OpTypes::kCastType), options_(options) {}

void CastTypeOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void CastTypeOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CastTypeOp>(shared_from_this(), i_irs, o_irs);
}

void CastTypeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("CastTypeOp::forward not implemented in aops base.");
}

void CastTypeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), options_.dtype, i.device()));
}

void CastTypeOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops