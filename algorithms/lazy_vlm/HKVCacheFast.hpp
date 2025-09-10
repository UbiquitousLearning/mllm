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
//
// This KVCache is managed in Paged method
//
// [B, S, H, D] is the format that received.
class HKVCacheFast {
 public:
 private:
  int32_t q_heads_;
  int32_t kv_heads_;
  int32_t kv_dims_;
  int32_t max_cache_length_;
  mllm::DeviceTypes device_type_;
  mllm::DataTypes k_dtype_;
  mllm::DataTypes v_dtype_;

  std::vector<mllm::Tensor> k_cache_;
  std::vector<mllm::Tensor> v_cache_;
  std::vector<mllm::Tensor> h_cache_;
};
