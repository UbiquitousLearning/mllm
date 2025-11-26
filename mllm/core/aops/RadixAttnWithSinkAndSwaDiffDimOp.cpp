// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/RadixAttnWithSinkAndSwaDiffDimOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

RadixAttnSwaSinkOp::RadixAttnSwaSinkOp(const RadixAttnSwaSinkOptions& options)
    : BaseOp(OpTypes::kRadixAttnWithSinkAndSwaDiffDim), options_(options) {}

void RadixAttnSwaSinkOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void RadixAttnSwaSinkOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::RadixAttnSwaSinkOp>(shared_from_this(), i_irs, o_irs);
}

void RadixAttnSwaSinkOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("RadixAttnSwaSinkOp::forward not implemented in aops base.");
}

void RadixAttnSwaSinkOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // CHECK sliding window is set
  MLLM_RT_ASSERT(options_.sliding_window != -1);

  // CHECK inputs
  MLLM_RT_ASSERT(inputs.size() >= 3);
  // CHECK QKV [B, S, H, D]
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  MLLM_RT_ASSERT(q.rank() == 4 && k.rank() == 1 && v.rank() == 1);
  if (options_.s_aux_enable) {
    MLLM_RT_ASSERT_EQ(inputs.size(), 4);
    auto& s_aux = inputs[3];
    MLLM_RT_ASSERT_EQ(s_aux.rank(), 1);
    MLLM_RT_ASSERT_EQ(s_aux.size(0), q.size(2));
  }

  outputs.emplace_back(mllm::Tensor::empty({options_.B, q.size(1), options_.q_head, options_.D_V}, q.dtype(), q.device()));
}

void RadixAttnSwaSinkOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
