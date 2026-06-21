// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/lmcache/DynamicCache.hpp"
#include "mllm/nn/Functional.hpp"

namespace mllm::nn {

DynamicCache::DynamicCache(int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims, bool use_fa2)
    : layer_nums_(layer_nums), q_heads_(q_heads), kv_heads_(kv_heads), kv_dims_(kv_dims), use_fa2_(use_fa2) {
  for (int i = 0; i < layer_nums_; ++i) {
    k_cache_.emplace_back(Tensor::nil());
    v_cache_.emplace_back(Tensor::nil());
  }
}

std::array<Tensor, 2> DynamicCache::updateKVCache(int32_t layer_idx, Tensor k, Tensor v) {
  // The input should be [B, H, S, D]
  MLLM_RT_ASSERT_EQ(k.shape()[1], kv_heads_);
  MLLM_RT_ASSERT_EQ(v.shape()[1], kv_heads_);

  if (!use_fa2_) {
    k = k.repeat(q_heads_ / kv_heads_, 1);
    v = v.repeat(q_heads_ / kv_heads_, 1);
  }

  if ((!k_cache_[layer_idx]) && (!v_cache_[layer_idx])) {
    k_cache_[layer_idx] = k;
    v_cache_[layer_idx] = v;
    return {k, v};
  }

  k_cache_[layer_idx] = functional::concat({k_cache_[layer_idx], k}, 2);
  v_cache_[layer_idx] = functional::concat({v_cache_[layer_idx], v}, 2);
  return {k_cache_[layer_idx], v_cache_[layer_idx]};
}

int32_t DynamicCache::getCurrentSeqCnt() const {
  if (k_cache_.empty() || !k_cache_[0]) { return 0; }
  return k_cache_[0].shape()[2];
}

}  // namespace mllm::nn