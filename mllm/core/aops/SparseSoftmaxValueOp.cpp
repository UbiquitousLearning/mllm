// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/SparseSoftmaxValueOp.hpp"

#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

SparseSoftmaxValueOp::SparseSoftmaxValueOp(const SparseSoftmaxValueOpOptions& options)
    : BaseOp(OpTypes::kSparseSoftmaxValue), options_(options) {}

void SparseSoftmaxValueOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void SparseSoftmaxValueOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::SparseSoftmaxValueOp>(shared_from_this(), i_irs, o_irs);
}

void SparseSoftmaxValueOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("SparseSoftmaxValueOp::forward not implemented in aops base.");
}

void SparseSoftmaxValueOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 2);
  const auto& logits = inputs[0];
  const auto& value = inputs[1];
  const auto& logits_shape = logits.shape();
  const auto& value_shape = value.shape();

  MLLM_RT_ASSERT_EQ(logits_shape.size(), 4);
  MLLM_RT_ASSERT_EQ(value_shape.size(), 4);
  MLLM_RT_ASSERT(logits_shape[0] > 0 && logits_shape[1] > 0 && logits_shape[2] > 0 && logits_shape[3] >= logits_shape[2]);
  MLLM_RT_ASSERT(value_shape[1] > 0 && value_shape[3] > 0);
  MLLM_RT_ASSERT_EQ(logits_shape[0], value_shape[0]);
  MLLM_RT_ASSERT_EQ(logits_shape[3], value_shape[2]);
  MLLM_RT_ASSERT_EQ(logits_shape[1] % value_shape[1], 0);
  MLLM_RT_ASSERT_EQ(logits.dtype(), kFloat32);
  MLLM_RT_ASSERT(value.dtype() == kFloat32 || value.dtype() == kFloat16);
  MLLM_RT_ASSERT_EQ(logits.device(), value.device());
  MLLM_RT_ASSERT(options_.sparsity >= 0.0F && options_.sparsity < 1.0F);
  MLLM_RT_ASSERT(options_.dense_tokens >= 0);
  MLLM_RT_ASSERT(options_.head_retentions.empty() || options_.head_retentions.size() == static_cast<size_t>(logits_shape[1]));
  for (const float retention : options_.head_retentions) { MLLM_RT_ASSERT(retention > 0.0F && retention <= 1.0F); }

  outputs.emplace_back(
      Tensor::empty({logits_shape[0], logits_shape[1], logits_shape[2], value_shape[3]}, kFloat32, logits.device()));
}

void SparseSoftmaxValueOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
