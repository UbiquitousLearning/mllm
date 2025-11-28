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

  const nn::StaticCache* cache_to_use = shared_cache_ ? shared_cache_ : &cache_;
  // When using own cache (not shared), cache_ has only 1 layer, so layer_idx should be 0
  // When using shared cache, use options_.layer_idx
  int32_t layer_idx_to_use = shared_cache_ ? options_.layer_idx : 0;

  // inputs[0] is k tensor, inputs[1] is v tensor
  // outputs[0] is updated k tensor, outputs[1] is updated v tensor
  outputs.emplace_back(Tensor::empty({B, options_.kv_head, S + cache_to_use->getCurrentSeqCnt(layer_idx_to_use), D}));
  outputs.emplace_back(Tensor::empty({B, options_.kv_head, S + cache_to_use->getCurrentSeqCnt(layer_idx_to_use), D}));
}

void CPUKVCacheOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // This KVCache Op is only for TRACE MODE to use.

  // inputs[0] is k tensor, inputs[1] is v tensor
  // outputs[0] is updated k tensor, outputs[1] is updated v tensor
  MLLM_RT_ASSERT_EQ(inputs.size(), 2U);
  MLLM_RT_ASSERT_EQ(outputs.size(), 2U);

  auto& k = inputs[0];
  auto& v = inputs[1];

  nn::StaticCache* cache_to_use = shared_cache_ ? shared_cache_ : &cache_;
  int32_t layer_idx_to_use = shared_cache_ ? options_.layer_idx : 0;

  // Update the KV cache and get the updated cache tensors
  auto [updated_k, updated_v] = cache_to_use->updateKVCache(layer_idx_to_use, k, v);

  // Copy the results to outputs
  outputs[0] = std::move(updated_k);
  outputs[1] = std::move(updated_v);
}

void CPUKVCacheOp::clearCache() {
  if (shared_cache_) {
    shared_cache_->clearCache();
  } else {
    cache_.clearCache();
  }
}

void CPUKVCacheOp::setCurrentSeqCnt(int32_t seq) {
  if (shared_cache_) {
    shared_cache_->setCurrentSeqCnt(seq);
  } else {
    cache_.setCurrentSeqCnt(seq);
  }
}

int32_t CPUKVCacheOp::getCurrentSeqCnt() const {
  int32_t layer_idx_to_use = shared_cache_ ? options_.layer_idx : 0;

  if (shared_cache_) {
    return shared_cache_->getCurrentSeqCnt(layer_idx_to_use);
  } else {
    return cache_.getCurrentSeqCnt(layer_idx_to_use);
  }
}

}  // namespace mllm::cpu
