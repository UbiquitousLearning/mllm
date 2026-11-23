// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/PatternSparseAttentionOp.hpp"

#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

PatternSparseAttentionOp::PatternSparseAttentionOp(const PatternSparseAttentionOpOptions& options)
    : BaseOp(OpTypes::kPatternSparseAttention), options_(options) {}

void PatternSparseAttentionOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void PatternSparseAttentionOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::PatternSparseAttentionOp>(shared_from_this(), i_irs, o_irs);
}

void PatternSparseAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("PatternSparseAttentionOp::forward not implemented in aops base.");
}

void PatternSparseAttentionOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const size_t expected_inputs = options_.hmx_selector ? 4 : 3;
  MLLM_RT_ASSERT_EQ(inputs.size(), expected_inputs);
  const auto& query = inputs[0];
  const auto& key = inputs[1];
  const auto& value = inputs[2];
  const auto& q_shape = query.shape();
  const auto& k_shape = key.shape();
  const auto& v_shape = value.shape();

  MLLM_RT_ASSERT_EQ(q_shape.size(), 4);
  MLLM_RT_ASSERT_EQ(k_shape.size(), 4);
  MLLM_RT_ASSERT_EQ(v_shape.size(), 4);
  MLLM_RT_ASSERT(q_shape[0] > 0 && q_shape[1] > 0 && q_shape[2] > 0 && q_shape[3] > 0);
  MLLM_RT_ASSERT(k_shape[1] > 0 && k_shape[2] >= q_shape[2] && v_shape[3] > 0);
  MLLM_RT_ASSERT_EQ(q_shape[0], k_shape[0]);
  MLLM_RT_ASSERT_EQ(q_shape[0], v_shape[0]);
  MLLM_RT_ASSERT_EQ(k_shape[1], v_shape[1]);
  MLLM_RT_ASSERT_EQ(k_shape[2], v_shape[2]);
  MLLM_RT_ASSERT_EQ(q_shape[3], k_shape[3]);
  MLLM_RT_ASSERT_EQ(q_shape[1] % k_shape[1], 0);
  MLLM_RT_ASSERT_EQ(query.dtype(), kFloat32);
  MLLM_RT_ASSERT(key.dtype() == kFloat32 || key.dtype() == kFloat16);
  MLLM_RT_ASSERT(value.dtype() == kFloat32 || value.dtype() == kFloat16);
  MLLM_RT_ASSERT_EQ(query.device(), key.device());
  MLLM_RT_ASSERT_EQ(query.device(), value.device());
  if (options_.hmx_selector) {
    const auto& selector_key = inputs[3];
    MLLM_RT_ASSERT_EQ(selector_key.shape(), k_shape);
    MLLM_RT_ASSERT_EQ(selector_key.dtype(), kFloat16);
    MLLM_RT_ASSERT_EQ(selector_key.device(), query.device());
  }

  MLLM_RT_ASSERT(options_.sparsity >= 0.0F && options_.sparsity < 1.0F);
  MLLM_RT_ASSERT(options_.local_ratio >= 0.0F && options_.local_ratio <= 1.0F);
  MLLM_RT_ASSERT(options_.prefix_tokens >= 0 && options_.dense_tokens >= 0 && options_.pack_reserve_tokens >= 0);
  MLLM_RT_ASSERT(!(options_.random_pattern && options_.hmx_selector));
  MLLM_RT_ASSERT(options_.head_retentions.empty() || options_.head_retentions.size() == static_cast<size_t>(q_shape[1]));
  for (const float retention : options_.head_retentions) { MLLM_RT_ASSERT(retention >= 0.0F && retention <= 1.0F); }

  outputs.emplace_back(Tensor::empty({q_shape[0], q_shape[1], q_shape[2], v_shape[3]}, kFloat32, query.device()));
}

void PatternSparseAttentionOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
