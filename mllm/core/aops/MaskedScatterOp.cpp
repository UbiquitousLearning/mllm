// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/MaskedScatterOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

MaskedScatterOp::MaskedScatterOp(const MaskedScatterOpOptions& options) : BaseOp(OpTypes::kMaskedScatter), options_(options) {}

void MaskedScatterOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void MaskedScatterOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::MaskedScatterOp>(shared_from_this(), i_irs, o_irs);
}

void MaskedScatterOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("MaskedScatterOp::forward not implemented in aops base.");
}

void MaskedScatterOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs.emplace_back(inputs[0]);
}

void MaskedScatterOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Do nothing.
  MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::aops
