// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/SparseSoftmaxValueOp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "mllm/backends/cpu/kernels/common/shadownpu/AttentionProfiler.hpp"
#include "mllm/backends/cpu/kernels/common/shadownpu/SparseSoftmaxValue.hpp"

namespace mllm::cpu {
namespace {

std::array<int32_t, 4> sparseSoftmaxValueStride(const Tensor& tensor) {
  const auto stride = tensor.stride();
  return {stride[0], stride[1], stride[2], stride[3]};
}

}  // namespace

void CPUSparseSoftmaxValueOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  shadownpu::ScopedAttentionProfile sparse_total_profile(shadownpu::AttentionProfileStage::SPARSE_TOTAL,
                                                         shadownpu::CPUAttentionProfiler::enabled());
  const auto logits_shape = inputs[0].shape();
  const auto value_shape = inputs[1].shape();
  shadownpu::SparseSoftmaxValueF32Params params;
  params.batch = logits_shape[0];
  params.query_heads = logits_shape[1];
  params.kv_heads = value_shape[1];
  params.query_len = logits_shape[2];
  params.key_len = logits_shape[3];
  params.value_dim = value_shape[3];
  params.logits_stride = sparseSoftmaxValueStride(inputs[0]);
  params.value_stride = sparseSoftmaxValueStride(inputs[1]);
  params.output_stride = sparseSoftmaxValueStride(outputs[0]);
  params.sparsity = options_.sparsity;
  params.causal_mask = options_.causal_mask;
  params.dense_tokens = options_.dense_tokens;
  params.head_retentions = options_.head_retentions.empty() ? nullptr : options_.head_retentions.data();

  uint64_t eligible = 0;
  uint64_t retained = 0;
  for (int32_t batch = 0; batch < params.batch; ++batch) {
    for (int32_t head = 0; head < params.query_heads; ++head) {
      const float density = params.head_retentions == nullptr ? 1.0F - params.sparsity : params.head_retentions[head];
      for (int32_t query_index = 0; query_index < params.query_len; ++query_index) {
        const int32_t valid = params.causal_mask && params.query_len > 1
                                  ? std::min(params.key_len, params.key_len - params.query_len + query_index + 1)
                                  : params.key_len;
        const int32_t keep =
            valid <= params.dense_tokens
                ? valid
                : std::max(1, std::min(valid, static_cast<int32_t>(std::ceil(density * static_cast<float>(valid)))));
        eligible += static_cast<uint64_t>(valid);
        retained += static_cast<uint64_t>(keep);
      }
    }
  }
  shadownpu::CPUSparseSelectionStats::add(eligible, retained);

  bool succeeded = false;
  if (inputs[1].dtype() == kFloat32) {
    succeeded =
        shadownpu::sparseSoftmaxValueTyped(params, inputs[0].ptr<float>(), inputs[1].ptr<float>(), outputs[0].ptr<float>());
  } else if (inputs[1].dtype() == kFloat16) {
    succeeded = shadownpu::sparseSoftmaxValueTyped(params, inputs[0].ptr<float>(), inputs[1].ptr<mllm_fp16_t>(),
                                                   outputs[0].ptr<float>());
  }
  if (!succeeded) { throw std::runtime_error("SparseSoftmaxValue CPU reference rejected its inputs or options"); }
}

}  // namespace mllm::cpu
