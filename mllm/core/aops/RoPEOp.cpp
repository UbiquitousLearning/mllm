// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/RoPEOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

RoPEOp::RoPEOp(const RoPEOpOptions& options) : BaseOp(OpTypes::kRoPE), options_(options) {}

void RoPEOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void RoPEOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::RoPEOp>(shared_from_this(), i_irs, o_irs);
}

void RoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("RoPEOp::forward not implemented in aops base.");
}

void RoPEOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // RoPE requires 3 inputs:
  // Pos 0: activations [B, H, S, D]
  // Pos 1: sin [S, D]
  // Pos 2: cos [S, D]
  // Output: [B, H, S, D]
  if (options_.isInplace()) {
    outputs.emplace_back(inputs[0]);
  } else {
    MLLM_RT_ASSERT_EQ(inputs.size(), 3);
    MLLM_RT_ASSERT_EQ(inputs[0].shape().size(), 4);

    outputs.emplace_back(Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device()));
  }
}

void RoPEOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (!options_.isInplace()) { BaseOp::setup(inputs, outputs); }
}

}  // namespace mllm::aops
