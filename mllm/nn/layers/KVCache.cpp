// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/core/aops/KVCacheOp.hpp"
#include "mllm/nn/layers/KVCache.hpp"

namespace mllm::nn {

KVCache::KVCache() : Layer(OpTypes::kKVCache, aops::KVCacheOpOptions{}) {}

KVCache::KVCache(const aops::KVCacheOpOptions& options) : Layer(OpTypes::kKVCache, options) {}

KVCache::KVCache(int32_t layer_idx, int32_t q_head, int32_t kv_head, int32_t head_dim, bool use_fa2)
    : Layer(OpTypes::kKVCache,
            aops::KVCacheOpOptions{
                .layer_idx = layer_idx, .q_head = q_head, .kv_head = kv_head, .head_dim = head_dim, .use_fa2 = use_fa2}) {}

void KVCache::setLayerIndex(int32_t layer_idx) {
  std::static_pointer_cast<aops::KVCacheOp>(impl()->getInstancedOp())->setLayerIndex(layer_idx);
}

void KVCache::clearCache() { std::static_pointer_cast<aops::KVCacheOp>(impl()->getInstancedOp())->clearCache(); }

void KVCache::setCurrentSeqCnt(int32_t seq) {
  std::static_pointer_cast<aops::KVCacheOp>(impl()->getInstancedOp())->setCurrentSeqCnt(seq);
}

int32_t KVCache::getCurrentSeqCnt() const {
  return std::static_pointer_cast<aops::KVCacheOp>(impl()->getInstancedOp())->getCurrentSeqCnt();
}

}  // namespace mllm::nn
