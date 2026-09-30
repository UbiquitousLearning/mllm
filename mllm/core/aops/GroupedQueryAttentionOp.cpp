// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/GroupedQueryAttentionOp.hpp"

#include <stdexcept>

#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

GroupedQueryAttentionOp::GroupedQueryAttentionOp(const GroupedQueryAttentionOpOptions& options)
    : BaseOp(OpTypes::kGroupedQueryAttention), options_(options) {}

void GroupedQueryAttentionOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void GroupedQueryAttentionOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::GroupedQueryAttentionOp>(shared_from_this(), i_irs, o_irs);
}

void GroupedQueryAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("GroupedQueryAttentionOp::forward not implemented in aops base.");
}

void GroupedQueryAttentionOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.sliding_window < 0
      || (options_.sliding_window != 0 && options_.implementation != GroupedQueryAttentionImplementation::kDirectStrided))
    throw std::invalid_argument("Sliding window requires DirectStrided and a nonnegative window");
  if (inputs.size() != 3) { throw std::invalid_argument("GroupedQueryAttention expects query, key, and value"); }
  const auto& query = inputs[0];
  const auto& key = inputs[1];
  const auto& value = inputs[2];
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 || q_shape[0] <= 0 || q_shape[0] != k_shape[0]
      || q_shape[0] != v_shape[0] || q_shape[1] <= 0 || k_shape[1] <= 0 || k_shape[1] != v_shape[1]
      || q_shape[1] % k_shape[1] != 0 || q_shape[2] <= 0 || k_shape[2] <= 0 || k_shape[2] != v_shape[2]
      || k_shape[2] < q_shape[2] || q_shape[3] <= 0 || q_shape[3] != k_shape[3] || v_shape[3] <= 0) {
    throw std::invalid_argument("GroupedQueryAttention expects compatible [B, H, S, D] tensors");
  }
  if (query.dtype() != key.dtype() || query.dtype() != value.dtype()
      || (query.dtype() != kFloat32 && query.dtype() != kFloat16)) {
    throw std::invalid_argument("GroupedQueryAttention requires matching float32 or float16 inputs");
  }
  if (query.device() != key.device() || query.device() != value.device()) {
    throw std::invalid_argument("GroupedQueryAttention inputs must be on the same device");
  }
  if (options_.implementation == GroupedQueryAttentionImplementation::kDecodeNativeKV) {
    if (q_shape[2] != 1) {
      throw std::invalid_argument("GroupedQueryAttention DecodeNativeKV accepts a single query position only");
    }
    if (query.dtype() != kFloat32) {
      throw std::invalid_argument("GroupedQueryAttention DecodeNativeKV supports float32 only");
    }
  }
  outputs.emplace_back(Tensor::empty({q_shape[0], q_shape[1], q_shape[2], v_shape[3]}, value.dtype(), value.device()));
}

void GroupedQueryAttentionOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
