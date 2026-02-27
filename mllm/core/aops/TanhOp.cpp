// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/TanhOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

TanhOp::TanhOp(const TanhOpOptions& options) : BaseOp(OpTypes::kTanh), options_(options) {}

void TanhOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void TanhOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::TanhOp>(shared_from_this(), i_irs, o_irs);
}

void TanhOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("TanhOp::forward not implemented in aops base.");
}

void TanhOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isInplace()) {
    outputs.emplace_back(inputs[0]);
  } else {
    outputs.emplace_back(Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device()));
  }
}

void TanhOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
