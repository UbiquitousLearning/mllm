// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cstdint>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"

namespace mllm::ascend {

/// Ascend-specific KV Cache that avoids slice+copy issues.
/// This cache stores KV tensors contiguously and uses simple memcpy for updates.
/// GQA repeat is NOT done here - it should be done in the attention computation.
class AscendKVCache {
 public:
  AscendKVCache() = default;

  /// Initialize the KV cache.
  /// @param max_cache_length Maximum sequence length to cache
  /// @param layer_nums Number of layers
  /// @param kv_heads Number of KV heads (not q_heads)
  /// @param head_dim Head dimension
  /// @param dtype Data type (typically FP16 for Ascend)
  /// @param num_key_value_groups Number of query head groups per KV head (for GQA)
  AscendKVCache(int32_t max_cache_length, int32_t layer_nums, int32_t kv_heads, int32_t head_dim,
                DataTypes dtype = kFloat16, int32_t num_key_value_groups = 1);

  /// Update KV cache with new key/value states.
  /// @param layer_idx Layer index
  /// @param k New key states [B, kv_heads, S, D]
  /// @param v New value states [B, kv_heads, S, D]
  /// @return Pair of (k_cached, v_cached) with full cached sequence [B, kv_heads, S, D]
  std::array<Tensor, 2> updateKVCache(int32_t layer_idx, const Tensor& k, const Tensor& v);

  /// Get current cached sequence length for a layer
  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const { return current_seq_cnt_[layer_idx]; }

  /// Advance cached sequence length after a graph/plugin has updated the cache buffers.
  void advanceSeqCnt(int32_t layer_idx, int32_t append_seq_len);

  /// Get number of layers
  [[nodiscard]] int32_t getLayerNums() const { return layer_nums_; }

  [[nodiscard]] int32_t getMaxCacheLength() const { return max_cache_length_; }

  /// Clear all cached data
  void clearCache();

  /// Get K cache buffer for a layer (for debugging)
  [[nodiscard]] Tensor getKCacheBuffer(int32_t layer_idx) const { return k_cache_[layer_idx]; }

  /// Get V cache buffer for a layer (for debugging)
  [[nodiscard]] Tensor getVCacheBuffer(int32_t layer_idx) const { return v_cache_[layer_idx]; }

 private:
  int32_t max_cache_length_{0};
  int32_t layer_nums_{0};
  int32_t kv_heads_{0};
  int32_t head_dim_{0};
  DataTypes dtype_{kFloat16};

  // k_cache_[layer]: [1, kv_heads, max_cache_length, head_dim] - contiguous on Ascend
  std::vector<Tensor> k_cache_;
  // v_cache_[layer]: [1, kv_heads, max_cache_length, head_dim] - contiguous on Ascend
  std::vector<Tensor> v_cache_;

  // Current sequence count per layer
  std::vector<int32_t> current_seq_cnt_;
};

/// Repeat interleave operation for GQA.
/// Expands [B, kv_heads, S, D] to [B, q_heads, S, D] by repeating each head.
/// @param x Input tensor [B, kv_heads, S, D]
/// @param repeat_times Number of times to repeat each head (q_heads / kv_heads)
/// @return Output tensor [B, q_heads, S, D]
Tensor repeatInterleaveForGQA(const Tensor& x, int32_t repeat_times);

}  // namespace mllm::ascend
