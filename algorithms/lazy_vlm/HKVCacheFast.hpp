// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <mllm/core/DataTypes.hpp>
#include <mllm/core/DeviceTypes.hpp>
#include <mllm/core/Tensor.hpp>

// Hidden States
// Key & Value
// Cache
//
// This KVCache is managed in Paged method
//
// [B, S, H, D] is the format that received.
class HKVCacheFast {
 public:
  ~HKVCacheFast() = default;

  HKVCacheFast() = default;

  HKVCacheFast(int32_t max_cache_length, int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims,
               mllm::DataTypes k_dtype, mllm::DataTypes v_dtype, mllm::DeviceTypes device_type = mllm::kCPU);

  void updateKVCache(int32_t layer_idx, mllm::Tensor k, mllm::Tensor v, const std::vector<int32_t>& pos);

  std::array<mllm::Tensor, 2> getKVCache(int32_t layer_idx);

  void initHiddenStateCache(int32_t batch_size, int32_t seq_len, int32_t hs_dims);

  mllm::Tensor getHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos);

  void updateHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos, mllm::Tensor hs_cache);

  void manualCacheLengthUpdate(int32_t layer_idx, int32_t cache_length_to_add);

  void visitHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos);

  int getCurrentSeqCnt(int32_t layer_idx);

  // To record all KV positions
  std::vector<std::vector<int>> occupied_kv_pos_;

  // To record only vision positions
  std::vector<std::vector<int>> kv_not_filled_pos_;

 private:
  int32_t q_heads_;
  int32_t kv_heads_;
  int32_t kv_dims_;
  int32_t layer_nums_;
  int32_t max_cache_length_;
  mllm::DeviceTypes device_type_;
  mllm::DataTypes k_dtype_;
  mllm::DataTypes v_dtype_;

  std::vector<mllm::Tensor> k_cache_;
  std::vector<mllm::Tensor> v_cache_;
  std::vector<mllm::Tensor> h_cache_;
  std::vector<int32_t> current_seq_cnt_;
};
