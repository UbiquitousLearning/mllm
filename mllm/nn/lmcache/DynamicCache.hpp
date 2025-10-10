// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <string>

#include "mllm/core/Tensor.hpp"

namespace mllm::nn {

class DynamicCache {
 public:
  ~DynamicCache() = default;

  DynamicCache(int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims, bool use_fa2 = true);

  [[nodiscard]] int32_t getCurrentSeqCnt() const;

  std::array<Tensor, 2> updateKVCache(int32_t layer_idx, Tensor k, Tensor v);

 private:
  int32_t layer_nums_;
  int32_t q_heads_;
  int32_t kv_heads_;
  int32_t kv_dims_;
  bool use_fa2_;

  std::vector<Tensor> k_cache_;
  std::vector<Tensor> v_cache_;
};

}  // namespace mllm::nn
