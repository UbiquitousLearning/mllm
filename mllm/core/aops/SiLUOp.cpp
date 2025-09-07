// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/SiLUOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

SiLUOp::SiLUOp(const SiLUOpOptions& options) : BaseOp(OpTypes::kSiLU), options_(options) {}

void SiLUOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void SiLUOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::SiLUOp>(shared_from_this(), i_irs, o_irs);
}

void SiLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("SiLUOp::forward not implemented in aops base.");
}

void SiLUOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isInplace()) {
    outputs.emplace_back(inputs[0]);
  } else {
    const auto& i = inputs[0];
    outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
  }
}

void SiLUOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (!options_.isInplace()) { BaseOp::setup(inputs, outputs); }
}

}  // namespace mllm::aops
