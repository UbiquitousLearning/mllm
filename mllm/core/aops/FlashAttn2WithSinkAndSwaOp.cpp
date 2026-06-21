// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/FlashAttn2WithSinkAndSwaOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

FlashAttention2SwaSinkOp::FlashAttention2SwaSinkOp(const FlashAttention2SwaSinkOptions& options)
    : BaseOp(OpTypes::kFlashAttention2), options_(options) {}

void FlashAttention2SwaSinkOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void FlashAttention2SwaSinkOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::FlashAttention2SwaSinkOp>(shared_from_this(), i_irs, o_irs);
}

void FlashAttention2SwaSinkOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("FlashAttention2SwaSinkOp::forward not implemented in aops base.");
}

void FlashAttention2SwaSinkOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // CHECK sliding window is set
  MLLM_RT_ASSERT(options_.sliding_window != -1);

  // CHECK inputs
  MLLM_RT_ASSERT(inputs.size() >= 3);
  // CHECK QKV [B, S, H, D]
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  MLLM_RT_ASSERT(q.rank() == 4 && k.rank() == 4 && v.rank() == 4);
  MLLM_RT_ASSERT_EQ(k.size(2), v.size(2));
  MLLM_RT_ASSERT_EQ(q.size(-1), k.size(-1));
  if (options_.s_aux_enable) {
    MLLM_RT_ASSERT_EQ(inputs.size(), 4);
    auto& s_aux = inputs[3];
    MLLM_RT_ASSERT_EQ(s_aux.rank(), 1);
    MLLM_RT_ASSERT_EQ(s_aux.size(0), q.size(2));
  }

  outputs.emplace_back(mllm::Tensor::empty({v.size(0), q.size(1), q.size(2), v.size(3)}, v.dtype(), v.device()));
}

void FlashAttention2SwaSinkOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
