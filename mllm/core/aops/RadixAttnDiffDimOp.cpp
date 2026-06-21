// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/RadixAttnDiffDimOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

RadixAttnRelaxOp::RadixAttnRelaxOp(const RadixAttnRelaxOpOptions& options)
    : BaseOp(OpTypes::kRadixAttnRelax), options_(options) {}

void RadixAttnRelaxOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void RadixAttnRelaxOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::RadixAttnRelaxOp>(shared_from_this(), i_irs, o_irs);
}

void RadixAttnRelaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("RadixAttnRelaxOp::forward not implemented in aops base.");
}

void RadixAttnRelaxOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // CHECK inputs
  MLLM_RT_ASSERT(inputs.size() >= 3);
  // CHECK QKV [B, S, H, D]
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  MLLM_RT_ASSERT(q.rank() == 4 && k.rank() == 1 && v.rank() == 1);
  outputs.emplace_back(mllm::Tensor::empty({options_.B, q.size(1), options_.q_head, options_.D_V}, q.dtype(), q.device()));
}

void RadixAttnRelaxOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
