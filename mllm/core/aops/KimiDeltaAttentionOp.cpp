// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/KimiDeltaAttentionOp.hpp"

#include <cmath>
#include <stdexcept>

#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

KimiDeltaAttentionOp::KimiDeltaAttentionOp(const KimiDeltaAttentionOpOptions& options)
    : BaseOp(OpTypes::kKimiDeltaAttention), options_(options) {}

void KimiDeltaAttentionOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void KimiDeltaAttentionOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::KimiDeltaAttentionOp>(shared_from_this(), i_irs, o_irs);
}

void KimiDeltaAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("KimiDeltaAttentionOp::forward not implemented in aops base.");
}

void KimiDeltaAttentionOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (inputs.size() != 8) { throw std::invalid_argument("KimiDeltaAttention expects 8 input tensors"); }

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const auto& gate = inputs[3];
  const auto& beta = inputs[4];
  const auto& a_log = inputs[5];
  const auto& dt_bias = inputs[6];
  const auto& state = inputs[7];
  if (q.rank() != 4 || k.shape() != q.shape() || v.shape() != q.shape() || gate.shape() != q.shape()) {
    throw std::invalid_argument("KimiDeltaAttention q, k, v, and gate must share [B, S, H, D] shape");
  }
  const auto batch = q.shape()[0];
  const auto sequence = q.shape()[1];
  const auto heads = q.shape()[2];
  const auto dim = q.shape()[3];
  if (beta.shape() != Tensor::shape_t{batch, sequence, heads}) {
    throw std::invalid_argument("KimiDeltaAttention beta must have [B, S, H] shape");
  }
  if (a_log.numel() != static_cast<std::size_t>(heads) || dt_bias.numel() != static_cast<std::size_t>(heads) * dim) {
    throw std::invalid_argument("KimiDeltaAttention decay parameters must contain H and H * D values");
  }
  if (state.shape() != Tensor::shape_t{batch, heads, dim, dim}) {
    throw std::invalid_argument("KimiDeltaAttention state must have [B, H, D, D] shape");
  }
  for (const auto& input : inputs) {
    if (input.dtype() != kFloat32 || input.device() != q.device()) {
      throw std::invalid_argument("KimiDeltaAttention requires float32 inputs on one device");
    }
  }
  if (options_.safe_gate && (!std::isfinite(options_.lower_bound) || options_.lower_bound >= 0.0F)) {
    throw std::invalid_argument("KimiDeltaAttention safe-gate lower bound must be finite and negative");
  }

  outputs.emplace_back(Tensor::empty(q.shape(), q.dtype(), q.device()));
  if (options_.state_inplace) {
    outputs.emplace_back(state);
  } else {
    outputs.emplace_back(Tensor::empty(state.shape(), state.dtype(), state.device()));
  }
}

void KimiDeltaAttentionOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.state_inplace) {
    outputs[0].alloc();
  } else {
    BaseOp::setup(inputs, outputs);
  }
}

}  // namespace mllm::aops
