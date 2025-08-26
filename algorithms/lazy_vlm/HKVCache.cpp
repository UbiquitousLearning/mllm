// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>

#include "HKVCache.hpp"
#include <mllm/core/Tensor.hpp>
#include <mllm/utils/Common.hpp>
#include <mllm/utils/UnsafeMacros.hpp>

HKVCache::HKVCache(int32_t max_cache_length, int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims,
                   mllm::DataTypes k_dtype, mllm::DataTypes v_dtype, mllm::DeviceTypes device_type, bool use_fa2)
    : device_type_(device_type),
      k_dtype_(k_dtype),
      v_dtype_(v_dtype),
      max_cache_length_(max_cache_length),
      layer_nums_(layer_nums),
      q_heads_(q_heads),
      kv_heads_(kv_heads),
      kv_dims_(kv_dims),
      use_fa2_(use_fa2) {
  // Inputs must be [B, S, H, D]
  if (use_fa2_) {
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
  } else
  // Inputs must be [B, H, S, D]
  {
    for (int i = 0; i < layer_nums_; ++i) {
      k_cache_.emplace_back(mllm::Tensor::empty(
                                {
                                    1,
                                    q_heads_,
                                    max_cache_length_,
                                    kv_dims_,
                                },
                                k_dtype_, device_type)
                                .alloc());
      v_cache_.emplace_back(mllm::Tensor::empty(
                                {
                                    1,
                                    q_heads_,
                                    max_cache_length_,
                                    kv_dims_,
                                },
                                v_dtype_, device_type)
                                .alloc());
      current_seq_cnt_.push_back(0);
    }
  }
}

int32_t HKVCache::getCurrentSeqCnt(int32_t layer_idx) const { return current_seq_cnt_[layer_idx]; }

__MLLM_UNSAFE_OPT_BEGIN_O3
std::array<mllm::Tensor, 2> HKVCache::updateKVCache(int32_t layer_idx, mllm::Tensor k, mllm::Tensor v, KVCacheUpdateRule rule,
                                                    std::unordered_map<std::string, mllm::AnyValue>& args) {
  switch (rule) {
    case KVCacheUpdateRule::kAppend: {
      auto inputs_seq_len = k.shape()[2];
      auto repeat_times = q_heads_ / kv_heads_;
      for (int h = 0; h < kv_heads_; ++h) {
        for (int r = 0; r < repeat_times; ++r) {
          // clang-format off
          auto k_cache_ptr = k_cache_[layer_idx].offsettedPtr<mllm::mllm_byte_t>({0, h * repeat_times + r, current_seq_cnt_[layer_idx], 0});
          auto v_cache_ptr = v_cache_[layer_idx].offsettedPtr<mllm::mllm_byte_t>({0, h * repeat_times + r, current_seq_cnt_[layer_idx], 0});
          // clang-format on
          auto k_ptr = k.offsettedPtr<mllm::mllm_byte_t>({0, h, 0, 0});
          auto v_ptr = v.offsettedPtr<mllm::mllm_byte_t>({0, h, 0, 0});
          // Copy
          std::memcpy(k_cache_ptr, k_ptr, inputs_seq_len * kv_dims_ * bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
          std::memcpy(v_cache_ptr, v_ptr, inputs_seq_len * kv_dims_ * bytesOfType(v_dtype_) / lanesOfType(v_dtype_));
        }
      }
      // Update sequence length.
      current_seq_cnt_[layer_idx] += inputs_seq_len;
      return {
          k_cache_[layer_idx][{mllm::kAll, mllm::kAll, {mllm::kAll, current_seq_cnt_[layer_idx]}, mllm::kAll}],
          v_cache_[layer_idx][{mllm::kAll, mllm::kAll, {mllm::kAll, current_seq_cnt_[layer_idx]}, mllm::kAll}],
      };
    }
    case KVCacheUpdateRule::kInsert: {
      auto pos = args["pos"].get<std::vector<int32_t>>();
      auto inputs_seq_len = k.shape()[2];
      auto repeat_times = q_heads_ / kv_heads_;
      MLLM_RT_ASSERT_EQ(pos.size(), inputs_seq_len);
      for (int i = 0; i < inputs_seq_len; ++i) {
        for (int h = 0; h < kv_heads_; ++h) {
          for (int r = 0; r < repeat_times; ++r) {
            // clang-format off
            auto k_cache_ptr = k_cache_[layer_idx].offsettedPtr<mllm::mllm_byte_t>({0, h * repeat_times + r, pos[i], 0});
            auto v_cache_ptr = v_cache_[layer_idx].offsettedPtr<mllm::mllm_byte_t>({0, h * repeat_times + r, pos[i], 0});
            // clang-format on
            auto k_ptr = k.offsettedPtr<mllm::mllm_byte_t>({0, h, i, 0});
            auto v_ptr = v.offsettedPtr<mllm::mllm_byte_t>({0, h, i, 0});
            // Copy
            std::memcpy(k_cache_ptr, k_ptr, kv_dims_ * bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
            std::memcpy(v_cache_ptr, v_ptr, kv_dims_ * bytesOfType(v_dtype_) / lanesOfType(v_dtype_));
          }
        }
      }
      return {mllm::Tensor::nil(), mllm::Tensor::nil()};
    }
    case KVCacheUpdateRule::kQuery: {
      auto pos = args["pos"].get<std::vector<int32_t>>();
      // [B, S, H, D]
      return {
          k_cache_[layer_idx][{{mllm::kAll}, pos, {mllm::kAll}, {mllm::kAll}}],
          v_cache_[layer_idx][{{mllm::kAll}, pos, {mllm::kAll}, {mllm::kAll}}],
      };
    }
  }

  return {mllm::Tensor::nil(), mllm::Tensor::nil()};
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
void HKVCache::initHiddenStateCache(int32_t batch_size, int32_t seq_len, int32_t head_nums, int32_t hs_dims) {
  // All hidden states input shape should be [B, S, H, D]
  MLLM_RT_ASSERT(h_cache_.size() == 0);
  for (int i = 0; i < layer_nums_; ++i) {
    h_cache_.emplace_back(mllm::Tensor::empty({batch_size, seq_len, head_nums, hs_dims}, mllm::kFloat32, mllm::kCPU).alloc());
  }
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
mllm::Tensor HKVCache::getHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos) {
  // [B, S, H, D]
  return h_cache_[layer_idx][{{mllm::kAll}, pos, {mllm::kAll}, {mllm::kAll}}];
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
void HKVCache::updateHiddenStateCache(int32_t layer_idx, const std::vector<int32_t>& pos, mllm::Tensor hs_cache) {
  auto S = hs_cache.shape()[1];
  auto H = hs_cache.shape()[2];
  auto D = hs_cache.shape()[3];
  MLLM_RT_ASSERT_EQ(S, pos.size());
  for (int s = 0; s < S; ++s) {
    auto cache_ptr = h_cache_[layer_idx].offsettedPtr<float32_t>({0, pos[s], 0, 0});
    auto now_ptr = hs_cache.offsettedPtr<float32_t>({0, s, 0, 0});
    std::memcpy(cache_ptr, now_ptr, H * D * sizeof(float32_t));
  }
}
__MLLM_UNSAFE_OPT_END
