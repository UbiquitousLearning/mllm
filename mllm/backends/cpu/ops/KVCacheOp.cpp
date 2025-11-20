// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/backends/cpu/ops/KVCacheOp.hpp"

namespace mllm::cpu {

CPUKVCacheOp::CPUKVCacheOp(const aops::KVCacheOpOptions& options)
    : aops::KVCacheOp(options),
      cache_(1024, 1, options.q_head, options.kv_head, options.head_dim, kFloat32, kFloat32, kCPU, options.use_fa2) {}

void CPUKVCacheOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Input is always [B, H, S, D]
  const int B = inputs[0].shape()[0];
  const int S = inputs[0].shape()[2];
  const int D = inputs[0].shape()[3];
  const DataTypes dtype = inputs[0].dtype();

  // inputs[0] is k tensor, inputs[1] is v tensor
  // outputs[0] is updated k tensor, outputs[1] is updated v tensor
  outputs.emplace_back(Tensor::empty({B, options_.kv_head, S + cache_.getCurrentSeqCnt(options_.layer_idx), D}));
  outputs.emplace_back(Tensor::empty({B, options_.kv_head, S + cache_.getCurrentSeqCnt(options_.layer_idx), D}));
}

void CPUKVCacheOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // This KVCache Op is only for TRACE MODE to use.

  // inputs[0] is k tensor, inputs[1] is v tensor
  // outputs[0] is updated k tensor, outputs[1] is updated v tensor
  MLLM_RT_ASSERT_EQ(inputs.size(), 2U);
  MLLM_RT_ASSERT_EQ(outputs.size(), 2U);

  auto& k = inputs[0];
  auto& v = inputs[1];

  // Update the KV cache and get the updated cache tensors
  auto [updated_k, updated_v] = cache_.updateKVCache(options_.layer_idx, k, v);

  // Copy the results to outputs
  outputs[0] = std::move(updated_k);
  outputs[1] = std::move(updated_v);
}

void CPUKVCacheOp::clearCache() { cache_.clearCache(); }

void CPUKVCacheOp::setCurrentSeqCnt(int32_t seq) { cache_.setCurrentSeqCnt(seq); }

int32_t CPUKVCacheOp::getCurrentSeqCnt() const { return cache_.getCurrentSeqCnt(options_.layer_idx); }

}  // namespace mllm::cpu
