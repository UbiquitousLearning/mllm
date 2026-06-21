// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include <stdexcept>

#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/UnsafeMacros.hpp"

namespace mllm::nn {
StaticCache::StaticCache(int32_t max_cache_length, int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims,
                         DataTypes k_dtype, DataTypes v_dtype, DeviceTypes device_type, bool use_fa2)
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
      k_cache_.emplace_back(Tensor::zeros(
          {
              1,
              max_cache_length_,
              kv_heads_,
              kv_dims_,
          },
          k_dtype_, device_type));
      v_cache_.emplace_back(Tensor::zeros(
          {
              1,
              max_cache_length_,
              kv_heads_,
              kv_dims_,
          },
          v_dtype_, device_type));
      current_seq_cnt_.push_back(0);
    }
  } else
  // Inputs must be [B, H, S, D]
  {
    for (int i = 0; i < layer_nums_; ++i) {
      k_cache_.emplace_back(Tensor::zeros(
          {
              1,
              q_heads_,
              max_cache_length_,
              kv_dims_,
          },
          k_dtype_, device_type));
      v_cache_.emplace_back(Tensor::zeros(
          {
              1,
              q_heads_,
              max_cache_length_,
              kv_dims_,
          },
          v_dtype_, device_type));
      current_seq_cnt_.push_back(0);
    }
  }
}

int32_t StaticCache::getCurrentSeqCnt(int32_t layer_idx) const { return current_seq_cnt_[layer_idx]; }

__MLLM_UNSAFE_OPT_BEGIN_O3
std::array<Tensor, 2> StaticCache::updateKVCache(int32_t layer_idx, Tensor k, Tensor v) {
  if (use_fa2_) {
    // The input should be [B, S, H, D]
    MLLM_RT_ASSERT_EQ(k.shape()[2], kv_heads_);
    MLLM_RT_ASSERT_EQ(v.shape()[2], kv_heads_);

    // TODO
    NYI("StaticCache for FA2 is not implemented yet");

    return {
        k_cache_[layer_idx][{kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll, kAll}],
        v_cache_[layer_idx][{kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll, kAll}],
    };
  } else
  // Eager mode
  {
    // The input should be [B, H, S, D]
    MLLM_RT_ASSERT_EQ(k.shape()[1], kv_heads_);
    MLLM_RT_ASSERT_EQ(v.shape()[1], kv_heads_);

    auto inputs_seq_len = k.shape()[2];

    auto repeat_times = q_heads_ / kv_heads_;

    switch (device_type_) {
      case kCPU: {
        for (int h = 0; h < kv_heads_; ++h) {
          for (int r = 0; r < repeat_times; ++r) {
            // clang-format off
            auto k_cache_ptr = k_cache_[layer_idx].offsettedPtr<mllm_byte_t>({0, h * repeat_times + r, current_seq_cnt_[layer_idx], 0});
            auto v_cache_ptr = v_cache_[layer_idx].offsettedPtr<mllm_byte_t>({0, h * repeat_times + r, current_seq_cnt_[layer_idx], 0});
            // clang-format on
            auto k_ptr = k.offsettedPtr<mllm_byte_t>({0, h, 0, 0});
            auto v_ptr = v.offsettedPtr<mllm_byte_t>({0, h, 0, 0});
            // Copy
            std::memcpy(k_cache_ptr, k_ptr, inputs_seq_len * kv_dims_ * bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
            std::memcpy(v_cache_ptr, v_ptr, inputs_seq_len * kv_dims_ * bytesOfType(v_dtype_) / lanesOfType(v_dtype_));
          }
        }

        break;
      }
      default: {
        for (int h = 0; h < kv_heads_; ++h) {
          for (int r = 0; r < repeat_times; ++r) {
            // clang-format off
            k[{kAll, h, kAll, kAll}].copy2(k_cache_[layer_idx][{kAll, h * repeat_times + r, {current_seq_cnt_[layer_idx], current_seq_cnt_[layer_idx] + inputs_seq_len}, kAll}]);
            v[{kAll, h, kAll, kAll}].copy2(v_cache_[layer_idx][{kAll, h * repeat_times + r, {current_seq_cnt_[layer_idx], current_seq_cnt_[layer_idx] + inputs_seq_len}, kAll}]);
            // clang-format on
          }
        }
        break;
      }
    }

    // Update sequence length.
    current_seq_cnt_[layer_idx] += inputs_seq_len;

    return {
        k_cache_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
        v_cache_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
    };
  }

  return {Tensor::nil(), Tensor::nil()};
}
__MLLM_UNSAFE_OPT_END

std::array<Tensor, 2> StaticCache::getKVCache(int32_t layer_idx) {
  if (use_fa2_) {
    return {
        k_cache_[layer_idx][{kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll, kAll}],
        v_cache_[layer_idx][{kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll, kAll}],
    };
  } else
  // Eager mode
  {
    return {
        k_cache_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
        v_cache_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
    };
  }
}

std::array<Tensor, 2> StaticCache::preGetKVWriteLocation(int32_t layer_idx, int32_t s) {
  if (!use_fa2_) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "StaticCache::preGetKVWriteLocation is not supported in eager mode"); }

  auto new_shape = {1, s, kv_heads_ * kv_dims_};
  auto k = k_cache_[layer_idx][{kAll, {current_seq_cnt_[layer_idx], current_seq_cnt_[layer_idx] + s}, kAll, kAll}];
  auto v = v_cache_[layer_idx][{kAll, {current_seq_cnt_[layer_idx], current_seq_cnt_[layer_idx] + s}, kAll, kAll}];
  current_seq_cnt_[layer_idx] += s;

  // NOTE: Only for Batch = 1
  // We wrap this KV in low level API to provide a contiguous tensor view.
  auto k_storage = TensorStorage::create(new_shape, k_cache_[layer_idx].dtype(), k_cache_[layer_idx].device());
  k_storage->mem_type_ = TensorMemTypes::kManual;
  k_storage->ptr_ = k.ptr<void>();
  k_storage->custom_32bit_uuid_ = k_cache_[layer_idx].uuid();
  auto out_k = TensorViewImpl::create(new_shape, k_storage);

  auto v_storage = TensorStorage::create(new_shape, v_cache_[layer_idx].dtype(), v_cache_[layer_idx].device());
  v_storage->mem_type_ = TensorMemTypes::kManual;
  v_storage->ptr_ = v.ptr<void>();
  v_storage->custom_32bit_uuid_ = v_cache_[layer_idx].uuid();
  auto out_v = TensorViewImpl::create(new_shape, v_storage);

  return {Tensor(out_k), Tensor(out_v)};
}

int32_t SubStaticCache::getCurrentSeqCnt(int32_t layer_idx) const { return current_sub_seq_cnt_[layer_idx]; }

std::array<Tensor, 2> SubStaticCache::updateKVCache(int32_t layer_idx, Tensor k, Tensor v) {
  auto inputs_seq_len = k.shape()[2];

  if (current_sub_seq_cnt_[layer_idx] + inputs_seq_len > sub_max_cache_length_) {
    throw std::runtime_error("SubStaticCache: sequence length exceeds sub-cache capacity");
  }

  // The input should be [B, H, S, D]
  MLLM_RT_ASSERT_EQ(k.shape()[1], cache_.kv_heads_);
  MLLM_RT_ASSERT_EQ(v.shape()[1], cache_.kv_heads_);

  auto repeat_times = cache_.q_heads_ / cache_.kv_heads_;

  switch (cache_.device_type_) {
    case kCPU: {
      for (int h = 0; h < cache_.kv_heads_; ++h) {
        for (int r = 0; r < repeat_times; ++r) {
          // clang-format off
            auto k_cache_ptr = sub_k_cache_[layer_idx].offsettedPtr<mllm_byte_t>({0, h * repeat_times + r, current_sub_seq_cnt_[layer_idx], 0});
            auto v_cache_ptr = sub_v_cache_[layer_idx].offsettedPtr<mllm_byte_t>({0, h * repeat_times + r, current_sub_seq_cnt_[layer_idx], 0});
          // clang-format on
          auto k_ptr = k.offsettedPtr<mllm_byte_t>({0, h, 0, 0});
          auto v_ptr = v.offsettedPtr<mllm_byte_t>({0, h, 0, 0});
          // Copy
          std::memcpy(k_cache_ptr, k_ptr,
                      inputs_seq_len * cache_.kv_dims_ * bytesOfType(cache_.k_dtype_) / lanesOfType(cache_.k_dtype_));
          std::memcpy(v_cache_ptr, v_ptr,
                      inputs_seq_len * cache_.kv_dims_ * bytesOfType(cache_.v_dtype_) / lanesOfType(cache_.v_dtype_));
        }
      }

      break;
    }
    default: {
      for (int h = 0; h < cache_.kv_heads_; ++h) {
        for (int r = 0; r < repeat_times; ++r) {
          // clang-format off
            k[{kAll, h, kAll, kAll}].copy2(sub_k_cache_[layer_idx][{kAll, h * repeat_times + r, {current_sub_seq_cnt_[layer_idx], current_sub_seq_cnt_[layer_idx] + inputs_seq_len}, kAll}]);
            v[{kAll, h, kAll, kAll}].copy2(sub_v_cache_[layer_idx][{kAll, h * repeat_times + r, {current_sub_seq_cnt_[layer_idx], current_sub_seq_cnt_[layer_idx] + inputs_seq_len}, kAll}]);
          // clang-format on
        }
      }
      break;
    }
  }

  // Update sequence length.
  current_sub_seq_cnt_[layer_idx] += inputs_seq_len;

  return {
      sub_k_cache_[layer_idx][{kAll, kAll, {kAll, current_sub_seq_cnt_[layer_idx]}, kAll}],
      sub_v_cache_[layer_idx][{kAll, kAll, {kAll, current_sub_seq_cnt_[layer_idx]}, kAll}],
  };
}

}  // namespace mllm::nn
