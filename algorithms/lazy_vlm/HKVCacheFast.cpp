// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>
#include <mllm/core/Tensor.hpp>
#include <mllm/utils/Common.hpp>
#include <mllm/utils/UnsafeMacros.hpp>

#include "HKVCacheFast.hpp"

HKVCacheFast::HKVCacheFast(int32_t max_cache_length, int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims,
                           mllm::DataTypes k_dtype, mllm::DataTypes v_dtype, mllm::DeviceTypes device_type)
    : device_type_(device_type),
      k_dtype_(k_dtype),
      v_dtype_(v_dtype),
      max_cache_length_(max_cache_length),
      layer_nums_(layer_nums),
      q_heads_(q_heads),
      kv_heads_(kv_heads),
      kv_dims_(kv_dims) {
  // Inputs must be [B, S, H, D]
  for (int i = 0; i < layer_nums_; ++i) {
    k_cache_.emplace_back(mllm::Tensor::empty(
                              {
                                  1,
                                  max_cache_length_,
                                  kv_heads_,
                                  kv_dims_,
                              },
                              k_dtype_, device_type)
                              .alloc());
    v_cache_.emplace_back(mllm::Tensor::empty(
                              {
                                  1,
                                  max_cache_length_,
                                  kv_heads_,
                                  kv_dims_,
                              },
                              v_dtype_, device_type)
                              .alloc());
    current_seq_cnt_.push_back(0);
  }
}

__MLLM_UNSAFE_OPT_BEGIN_O3
void HKVCacheFast::updateKVCache(int32_t layer_idx, mllm::Tensor k, mllm::Tensor v, const std::vector<int32_t>& pos) {
  // Inputs is [B, S, H, D]
  auto k_s_len = k.shape()[1];
  auto v_s_len = v.shape()[1];
  MLLM_RT_ASSERT_EQ(k_s_len, v_s_len);

  // Get current length
  auto c_len = current_seq_cnt_[layer_idx];

  // There is no need to repeat many times at head dim.
  k.copy2(k_cache_[layer_idx][{{mllm::kAll}, {c_len, c_len + k_s_len}, {mllm::kAll}, {mllm::kAll}}]);
  v.copy2(v_cache_[layer_idx][{{mllm::kAll}, {c_len, c_len + v_s_len}, {mllm::kAll}, {mllm::kAll}}]);
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3
std::array<mllm::Tensor, 2> HKVCacheFast::getKVCache(int32_t layer_idx) { return {k_cache_[layer_idx], v_cache_[layer_idx]}; }
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3
void HKVCacheFast::initHiddenStateCache(int32_t batch_size, int32_t seq_len, int32_t hs_dims) {
  // All hidden states input shape should be [B, S, H, D]
  MLLM_RT_ASSERT(h_cache_.size() == 0);
  for (int i = 0; i < layer_nums_; ++i) {
    h_cache_.emplace_back(mllm::Tensor::empty({batch_size, seq_len, hs_dims}, mllm::kFloat32, mllm::kCPU).alloc());
  }
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3
mllm::Tensor HKVCacheFast::getHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos) {
  // [B, S, D]
  return h_cache_[layer_idx][{{mllm::kAll}, pos, {mllm::kAll}}];
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3
void HKVCacheFast::updateHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos, mllm::Tensor hs_cache) {
  MLLM_RT_ASSERT_EQ(hs_cache.rank(), 3);
  auto S = hs_cache.shape()[1];
  auto D = hs_cache.shape()[2];
  MLLM_RT_ASSERT_EQ(S, pos.size());
  for (int s = 0; s < S; ++s) {
    auto cache_ptr = h_cache_[layer_idx].offsettedPtr<float32_t>({0, pos[s], 0});
    auto now_ptr = hs_cache.offsettedPtr<float32_t>({0, s, 0});
    std::memcpy(cache_ptr, now_ptr, D * sizeof(float32_t));
  }
}
__MLLM_UNSAFE_OPT_END
