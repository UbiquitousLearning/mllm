// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/FlashAttention2Op.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

FlashAttention2Op::FlashAttention2Op(const FlashAttention2OpOptions& options)
    : BaseOp(OpTypes::kFlashAttention2), options_(options) {}

void FlashAttention2Op::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void FlashAttention2Op::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::FlashAttention2Op>(shared_from_this(), i_irs, o_irs);
}

void FlashAttention2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("FlashAttention2Op::forward not implemented in aops base.");
}

void FlashAttention2Op::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void FlashAttention2Op::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops