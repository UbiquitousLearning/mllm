// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::nn {

class StaticCache {
 public:
  ~StaticCache() = default;

  StaticCache(int32_t max_cache_length, int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims,
              DataTypes k_dtype, DataTypes v_dtype, DeviceTypes device_type = kCPU, bool use_fa2 = true);

  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const;

  std::array<Tensor, 2> updateKVCache(int32_t layer_idx, Tensor k, Tensor v);

 private:
  DeviceTypes device_type_;
  DataTypes k_dtype_;
  DataTypes v_dtype_;
  int32_t max_cache_length_;
  int32_t layer_nums_;
  int32_t q_heads_;
  int32_t kv_heads_;
  int32_t kv_dims_;
  bool use_fa2_;

  std::vector<Tensor> k_cache_;
  std::vector<Tensor> v_cache_;
  std::vector<int32_t> current_seq_cnt_;
};

}  // namespace mllm::nn
