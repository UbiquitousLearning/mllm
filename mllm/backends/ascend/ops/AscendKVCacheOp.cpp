// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendKVCacheOp.hpp"

#include <acl/acl.h>

#include "mllm/utils/Common.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"

namespace mllm::ascend {

AscendKVCache::AscendKVCache(int32_t max_cache_length, int32_t layer_nums, int32_t kv_heads, int32_t head_dim,
                             DataTypes dtype, int32_t num_key_value_groups)
    : max_cache_length_(max_cache_length),
      layer_nums_(layer_nums),
      kv_heads_(kv_heads),
      head_dim_(head_dim),
      num_key_value_groups_(num_key_value_groups),
      dtype_(dtype) {
  // Allocate contiguous cache tensors for each layer
  // Shape: [1, kv_heads, max_cache_length, head_dim]
  for (int i = 0; i < layer_nums_; ++i) {
    k_cache_.emplace_back(
        Tensor::zeros({1, kv_heads_, max_cache_length_, head_dim_}, dtype_, kAscend));
    v_cache_.emplace_back(
        Tensor::zeros({1, kv_heads_, max_cache_length_, head_dim_}, dtype_, kAscend));
    current_seq_cnt_.push_back(0);
  }

  // For GQA: allocate repeated cache tensors
  if (num_key_value_groups_ > 1) {
    int32_t q_heads = kv_heads_ * num_key_value_groups_;
    for (int i = 0; i < layer_nums_; ++i) {
      k_cache_repeated_.emplace_back(
          Tensor::zeros({1, q_heads, max_cache_length_, head_dim_}, dtype_, kAscend));
      v_cache_repeated_.emplace_back(
          Tensor::zeros({1, q_heads, max_cache_length_, head_dim_}, dtype_, kAscend));
    }
  }
}

void AscendKVCache::clearCache() {
  for (int32_t layer_idx = 0; layer_idx < layer_nums_; ++layer_idx) {
    current_seq_cnt_[layer_idx] = 0;
  }
}

std::array<Tensor, 2> AscendKVCache::updateKVCache(int32_t layer_idx, const Tensor& k, const Tensor& v) {
  // Input k, v shape: [B, kv_heads, S, D]
  MLLM_RT_ASSERT_EQ(k.shape()[1], kv_heads_);
  MLLM_RT_ASSERT_EQ(v.shape()[1], kv_heads_);
  MLLM_RT_ASSERT(k.device() == kAscend && v.device() == kAscend);

  auto inputs_seq_len = k.shape()[2];
  auto batch_size = k.shape()[0];

  // Check if we have enough space
  if (current_seq_cnt_[layer_idx] + inputs_seq_len > max_cache_length_) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError,
                    "AscendKVCache: sequence length {} + {} exceeds max_cache_length {}",
                    current_seq_cnt_[layer_idx], inputs_seq_len, max_cache_length_);
  }

  // Calculate byte sizes
  const size_t element_size = bytesOfType(dtype_) / lanesOfType(dtype_);

  // Memory layout:
  // k/v input: [1, kv_heads, inputs_seq_len, head_dim] - stride between heads is inputs_seq_len * head_dim
  // k/v cache: [1, kv_heads, max_cache_length, head_dim] - stride between heads is max_cache_length * head_dim
  //
  // Since the strides differ, we must copy each head separately to place data at correct cache positions.

  const size_t head_seq_elements = inputs_seq_len * head_dim_;
  const size_t head_copy_bytes = head_seq_elements * element_size;

  // Stride between heads in source tensor (k/v)
  const size_t src_head_stride = inputs_seq_len * head_dim_ * element_size;
  // Stride between heads in cache tensor
  const size_t cache_head_stride = max_cache_length_ * head_dim_ * element_size;
  // Offset within each head to the current sequence position
  const size_t seq_offset = current_seq_cnt_[layer_idx] * head_dim_ * element_size;

  const char* k_src_base = static_cast<const char*>(k.ptr<void>());
  const char* v_src_base = static_cast<const char*>(v.ptr<void>());
  char* k_cache_base = static_cast<char*>(k_cache_[layer_idx].ptr<void>());
  char* v_cache_base = static_cast<char*>(v_cache_[layer_idx].ptr<void>());

  // Copy each head's data separately
  for (int32_t h = 0; h < kv_heads_; ++h) {
    const void* k_src = k_src_base + h * src_head_stride;
    const void* v_src = v_src_base + h * src_head_stride;
    void* k_dst = k_cache_base + h * cache_head_stride + seq_offset;
    void* v_dst = v_cache_base + h * cache_head_stride + seq_offset;

    auto ret = aclrtMemcpy(k_dst, head_copy_bytes, k_src, head_copy_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      MLLM_ACL_CHECK(ret);
    }

    ret = aclrtMemcpy(v_dst, head_copy_bytes, v_src, head_copy_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      MLLM_ACL_CHECK(ret);
    }
  }

  // CRITICAL: Sync required for long sequences to ensure KV cache is updated before next layer
  syncGlobalAtbStream();

  // Save old sequence count before updating (for incremental GQA repeat)
  const int32_t old_seq_cnt = current_seq_cnt_[layer_idx];

  // Update sequence count
  current_seq_cnt_[layer_idx] += inputs_seq_len;

  // ===== Optimization: GQA Incremental Update =====
  // If GQA is enabled, update repeated cache and return it
  if (num_key_value_groups_ > 1) {
    // Only copy the newly added tokens, not the entire history
    // This changes complexity from O(N²) to O(N) for generating N tokens
    const size_t element_size = bytesOfType(dtype_) / lanesOfType(dtype_);

    // NEW: Calculate size of only the new sequence portion
    const size_t new_seq_bytes = inputs_seq_len * head_dim_ * element_size;

    // NEW: Calculate offset to the new sequence portion in cache
    const size_t cache_seq_offset = old_seq_cnt * head_dim_ * element_size;

    const char* k_src_base = static_cast<const char*>(k_cache_[layer_idx].ptr<void>());
    const char* v_src_base = static_cast<const char*>(v_cache_[layer_idx].ptr<void>());
    char* k_dst_base = static_cast<char*>(k_cache_repeated_[layer_idx].ptr<void>());
    char* v_dst_base = static_cast<char*>(v_cache_repeated_[layer_idx].ptr<void>());

    const size_t src_head_stride = max_cache_length_ * head_dim_ * element_size;
    const size_t dst_head_stride = max_cache_length_ * head_dim_ * element_size;

    for (int32_t h = 0; h < kv_heads_; ++h) {
      // NEW: Source pointer now points to the new sequence portion (with offset)
      const void* k_src = k_src_base + h * src_head_stride + cache_seq_offset;
      const void* v_src = v_src_base + h * src_head_stride + cache_seq_offset;

      // Repeat this KV head to num_key_value_groups_ Q head positions
      for (int32_t r = 0; r < num_key_value_groups_; ++r) {
        // NEW: Destination pointer also points to the new sequence portion (with offset)
        void* k_dst = k_dst_base + (h * num_key_value_groups_ + r) * dst_head_stride + cache_seq_offset;
        void* v_dst = v_dst_base + (h * num_key_value_groups_ + r) * dst_head_stride + cache_seq_offset;

        // NEW: Only copy new_seq_bytes (not entire history)
        auto ret = aclrtMemcpy(k_dst, new_seq_bytes, k_src, new_seq_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
          MLLM_ACL_CHECK(ret);
        }

        ret = aclrtMemcpy(v_dst, new_seq_bytes, v_src, new_seq_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
          MLLM_ACL_CHECK(ret);
        }
      }
    }

    // CRITICAL: Sync required for GQA repeated cache update
    syncGlobalAtbStream();

    // Return sliced repeated cache [1, q_heads, current_seq, head_dim]
    return {
        k_cache_repeated_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
        v_cache_repeated_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
    };
  }

  // MHA case: return sliced cache tensors [1, kv_heads, current_seq, head_dim]
  return {
      k_cache_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
      v_cache_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
  };
}

Tensor repeatInterleaveForGQA(const Tensor& x, int32_t repeat_times) {
  // Input x: [B, kv_heads, S, D]
  // Output: [B, q_heads, S, D] where q_heads = kv_heads * repeat_times

  if (repeat_times == 1) {
    return x;  // No repeat needed (MHA case)
  }

  const auto& shape = x.shape();
  MLLM_RT_ASSERT_EQ(shape.size(), 4);

  int B = shape[0];
  int kv_heads = shape[1];
  int S = shape[2];
  int D = shape[3];
  int q_heads = kv_heads * repeat_times;

  // Create output tensor
  auto output = Tensor::empty({B, q_heads, S, D}, x.dtype(), x.device()).alloc();

  const size_t element_size = bytesOfType(x.dtype()) / lanesOfType(x.dtype());
  const size_t head_size = S * D * element_size;  // Size of one head's data

  // For each KV head, copy it repeat_times to consecutive positions in output
  for (int h = 0; h < kv_heads; ++h) {
    const size_t src_offset = h * head_size;
    const void* src_ptr = static_cast<const char*>(x.ptr<void>()) + src_offset;

    for (int r = 0; r < repeat_times; ++r) {
      const size_t dst_offset = (h * repeat_times + r) * head_size;
      void* dst_ptr = static_cast<char*>(output.ptr<void>()) + dst_offset;

      auto ret = aclrtMemcpy(dst_ptr, head_size, src_ptr, head_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
      if (ret != ACL_SUCCESS) {
        MLLM_ACL_CHECK(ret);
      }
    }
  }

  // CRITICAL: Sync required for GQA repeat operation
  syncGlobalAtbStream();

  return output;
}

}  // namespace mllm::ascend
