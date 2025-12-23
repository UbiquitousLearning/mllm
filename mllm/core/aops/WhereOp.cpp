// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/WhereOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

WhereOp::WhereOp(const WhereOpOptions& options) : BaseOp(OpTypes::kWhere), options_(options) {}

void WhereOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void WhereOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::WhereOp>(shared_from_this(), i_irs, o_irs);
}

void WhereOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("WhereOp::forward not implemented in aops base.");
}

void WhereOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs.emplace_back(Tensor::emptyLike(inputs[1]));
}

void WhereOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
