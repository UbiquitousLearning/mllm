// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include <mllm/core/DataTypes.hpp>
#include <mllm/core/DeviceTypes.hpp>
#include <mllm/core/Tensor.hpp>
#include "mllm/utils/AnyValue.hpp"

// Hidden States
// Key & Value
// Cache
class HKVCache {
 public:
  enum class KVCacheUpdateRule {
    kAppend = 0,  // Decoding
    kInsert = 1,  // Insert at which position. For Lazy
    kQuery = 2,   // Query KV Tensor from using a position vector.
  };

  ~HKVCache() = default;

  HKVCache() = default;

  HKVCache(int32_t max_cache_length, int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims,
           mllm::DataTypes k_dtype, mllm::DataTypes v_dtype, mllm::DeviceTypes device_type = mllm::kCPU, bool use_fa2 = true);

  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const;

  std::array<mllm::Tensor, 2> updateKVCache(int32_t layer_idx, mllm::Tensor k, mllm::Tensor v, KVCacheUpdateRule rule,
                                            const std::unordered_map<std::string, mllm::AnyValue>& args = {});

  void initHiddenStateCache(int32_t batch_size, int32_t seq_len, int32_t hs_dims);

  mllm::Tensor getHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos);

  void updateHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos, mllm::Tensor hs_cache);

  void visitHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos);

  void manualCacheLengthUpdate(int32_t layer_idx, int32_t times = 1);

  void reorderKVCache(int start = 0);

  std::vector<std::vector<int>> kv_not_filled_pos_;
  std::vector<std::vector<int>> visited_kv_pos_;

 private:
  mllm::DeviceTypes device_type_;
  mllm::DataTypes k_dtype_;
  mllm::DataTypes v_dtype_;
  int32_t max_cache_length_;
  int32_t layer_nums_;
  int32_t q_heads_;
  int32_t kv_heads_;
  int32_t kv_dims_;
  bool use_fa2_;

  std::vector<mllm::Tensor> k_cache_;
  std::vector<mllm::Tensor> v_cache_;
  std::vector<mllm::Tensor> h_cache_;
  std::vector<int32_t> current_seq_cnt_;
};
