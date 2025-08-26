// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <mllm/core/DataTypes.hpp>
#include <mllm/core/DeviceTypes.hpp>
#include <mllm/core/Tensor.hpp>

// Hidden States
// Key & Value
// Cache
class HKVCache {
 public:
  ~HKVCache() = default;

  HKVCache() = default;

  HKVCache(int32_t max_cache_length, int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims,
           mllm::DataTypes k_dtype, mllm::DataTypes v_dtype, mllm::DeviceTypes device_type = mllm::kCPU, bool use_fa2 = true);

  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const;

  std::array<mllm::Tensor, 2> updateKVCache(int32_t layer_idx, mllm::Tensor k, mllm::Tensor v);

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
  std::vector<int32_t> current_seq_cnt_;
};
