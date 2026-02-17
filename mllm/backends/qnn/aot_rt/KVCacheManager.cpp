// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include <cstring>
#include <algorithm>
#include "mllm/utils/Log.hpp"

namespace mllm::qnn::aot {

template<typename T>
KVCacheManager<T>::KVCacheManager(QnnAOTConfig config) : config_(config) {
  k_cache_.resize(config_.num_layers);
  v_cache_.resize(config_.num_layers);

  // Calculate cache size
  size_t cache_in_bytes = config_.num_layers * config_.num_heads * config_.head_dim * config_.max_cache_len * sizeof(T);
  size_t cache_out_bytes = config_.num_layers * config_.num_heads * config_.head_dim * config_.max_ar_len * sizeof(T);
  total_cache_size_ = 2 * (cache_in_bytes + cache_out_bytes);
}

template<typename T>
void KVCacheManager<T>::initCache(mllm::Allocator* allocator, int32_t ar_len) {
  cur_ar_len_ = ar_len;
  const size_t max_in_cache_block_in_bytes = config_.max_cache_len * sizeof(T);
  const size_t max_out_cache_block_in_bytes = config_.max_ar_len * sizeof(T);

  const size_t cache_in_bytes = config_.num_heads * config_.head_dim * max_in_cache_block_in_bytes;
  const size_t cache_out_bytes = config_.num_heads * config_.head_dim * max_out_cache_block_in_bytes;

  // Directly use Storage created by QNNAllocator
  // TODO: QNN shared buffer pool(custom mem) support
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    // Allocate buffer for key cache and value cache
    auto k_storage_in = std::make_shared<mllm::Storage>();
    k_storage_in->size_ = cache_in_bytes;
    allocator->alloc(k_storage_in);
    memset(k_storage_in->ptr_, 0, cache_in_bytes);

    auto k_storage_out = std::make_shared<mllm::Storage>();
    k_storage_out->size_ = cache_out_bytes;
    allocator->alloc(k_storage_out);
    memset(k_storage_out->ptr_, 0, cache_out_bytes);

    auto v_storage_in = std::make_shared<mllm::Storage>();
    v_storage_in->size_ = cache_in_bytes;
    allocator->alloc(v_storage_in);
    memset(v_storage_in->ptr_, 0, cache_in_bytes);

    auto v_storage_out = std::make_shared<mllm::Storage>();
    v_storage_out->size_ = cache_out_bytes;
    allocator->alloc(v_storage_out);
    memset(v_storage_out->ptr_, 0, cache_out_bytes);

    k_cache_[layer].buffer_storage = k_storage_in;
    k_cache_[layer].output_buffer_storage = k_storage_out;
    k_cache_[layer].buffer = reinterpret_cast<T*>(k_storage_in->ptr_);
    k_cache_[layer].output_buffer = reinterpret_cast<T*>(k_storage_out->ptr_);

    v_cache_[layer].buffer_storage = v_storage_in;
    v_cache_[layer].output_buffer_storage = v_storage_out;
    v_cache_[layer].buffer = reinterpret_cast<T*>(v_storage_in->ptr_);
    v_cache_[layer].output_buffer = reinterpret_cast<T*>(v_storage_out->ptr_);
  }
}

template<typename T>
void KVCacheManager<T>::initAttentionMask(uint16_t* attention_mask, const std::vector<int32_t>& attention_map, int32_t ar_len,
                                          int32_t n_past) {
  if (attention_map.size() > ar_len) {
    MLLM_ERROR("The size of attention_map ({}) doesn't match with ar_len ({})", attention_map.size(), ar_len);
    exit(1);
  }

  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * config_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  uint16_t* past_ptr = attention_mask;
  uint16_t* new_ptr = attention_mask + (config_.context_len - ar_len);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      uint16_t* parent_ptr = attention_mask + pidx * config_.context_len;
      std::memcpy(past_ptr, parent_ptr, config_.context_len * sizeof(uint16_t));
    }
    // Attend to itself
    new_ptr[i] = pos_val;
    past_ptr += config_.context_len;
    new_ptr += config_.context_len;
  }
}

template<typename T>
void KVCacheManager<T>::initAttentionMask(uint16_t* attention_mask, const std::vector<int32_t>& attention_map, int32_t ar_len,
                                          int32_t n_past, int32_t sliding_window, const std::vector<int32_t>& position_offset) {
  if (attention_map.size() > ar_len) {
    MLLM_ERROR("The size of attention_map ({}) doesn't match with ar_len ({})", attention_map.size(), ar_len);
    exit(1);
  }

  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * config_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  uint16_t* past_ptr = attention_mask;
  uint16_t* new_ptr = attention_mask + (config_.context_len - ar_len);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      uint16_t* parent_ptr = attention_mask + pidx * config_.context_len;
      std::memcpy(past_ptr, parent_ptr, config_.context_len * sizeof(uint16_t));
    }
    // Attend to itself
    new_ptr[i] = pos_val;

    // mask by limitation of sliding_window
    int32_t available_context_len =
        position_offset.empty() ? sliding_window - (i + 1) - n_past : sliding_window - (position_offset[i] + 1) - n_past;
    if (n_past > available_context_len) { std::fill_n(past_ptr, n_past - available_context_len, neg_val); }

    past_ptr += config_.context_len;
    new_ptr += config_.context_len;
  }
}

template<typename T>
void KVCacheManager<T>::updateAttentionMask(uint16_t* attention_mask, int32_t ar_len, int32_t n_past, int32_t n_update) {
  uint16_t pos_val = 65535;
  uint16_t* cur_ptr = attention_mask;
  cur_ptr += n_past;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    cur_ptr += config_.context_len;
  }
}

template<typename T>
void KVCacheManager<T>::updateAttentionMask(uint16_t* attention_mask, int32_t ar_len, int32_t n_past, int32_t n_update,
                                            int32_t sliding_window, const std::vector<int32_t>& position_offset) {
  uint16_t pos_val = 65535;
  uint16_t neg_val = 0;
  uint16_t* cur_ptr = attention_mask;
  cur_ptr += n_past;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    int32_t available_cache_len =
        position_offset.empty() ? sliding_window - (i + 1) : sliding_window - (position_offset[i] + 1);
    if (n_past + n_update > available_cache_len) {
      std::fill_n(cur_ptr - n_past, n_past + n_update - available_cache_len, neg_val);
    }
    cur_ptr += config_.context_len;
  }
}

template<typename T>
void KVCacheManager<T>::rearrangeCache(int32_t ar_len_dst) {
  // Don't need to rearrange if cur_ar_len_ is equal to target ar_len
  if (cur_ar_len_ == ar_len_dst) return;
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    rearrangeKey(k_cache_[layer], ar_len_dst);
    rearrangeValue(v_cache_[layer], ar_len_dst);
  }
  // rearrange done.
  cur_ar_len_ = ar_len_dst;
}

template<typename T>
void KVCacheManager<T>::rearrangeKey(KVCache<T>& k_cache, int32_t ar_len_dst) {
  // [B, H, D, S] rearrange.
  const int32_t src_cache_num = (cur_ar_len_ == config_.context_len) ? config_.context_len : config_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = config_.context_len - ar_len_dst;
  T* k_cache_in_read_ptr = k_cache.buffer;
  T* k_cache_in_write_ptr = k_cache.buffer;

  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < config_.head_dim * config_.num_heads; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, dst_cache_num * sizeof(T));
      k_cache_in_read_ptr += src_cache_num;
      k_cache_in_write_ptr += dst_cache_num;
    }
  } else {
    k_cache_in_read_ptr += (config_.head_dim * config_.num_heads - 1) * src_cache_num;
    k_cache_in_write_ptr += (config_.head_dim * config_.num_heads - 1) * dst_cache_num;
    // copy from last dimension
    for (int i = 0; i < config_.head_dim * config_.num_heads; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, src_cache_num * sizeof(T));
      k_cache_in_read_ptr -= src_cache_num;
      k_cache_in_write_ptr -= dst_cache_num;
    }
  }
}

template<typename T>
void KVCacheManager<T>::rearrangeValue(KVCache<T>& v_cache, int32_t ar_len_dst) {
  // [B, H, S, D] rearrange.
  const int32_t src_cache_num = (cur_ar_len_ == config_.context_len) ? config_.context_len : config_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = config_.context_len - ar_len_dst;
  T* v_cache_in_read_ptr = v_cache.buffer;
  T* v_cache_in_write_ptr = v_cache.buffer;
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < config_.num_heads; i++) {
      std::memmove(v_cache_in_write_ptr, v_cache_in_read_ptr, dst_cache_num * config_.head_dim * sizeof(T));
      v_cache_in_read_ptr += src_cache_num * config_.head_dim;
      v_cache_in_write_ptr += dst_cache_num * config_.head_dim;
    }
  } else {
    v_cache_in_read_ptr += config_.head_dim * (config_.num_heads - 1) * src_cache_num;
    v_cache_in_write_ptr += config_.head_dim * (config_.num_heads - 1) * dst_cache_num;
    // copy from last dimension
    for (int i = 0; i < config_.num_heads; i++) {
      std::memmove(v_cache_in_write_ptr, v_cache_in_read_ptr, src_cache_num * config_.head_dim * sizeof(T));
      v_cache_in_read_ptr -= src_cache_num * config_.head_dim;
      v_cache_in_write_ptr -= dst_cache_num * config_.head_dim;
    }
  }
}

template<typename T>
void KVCacheManager<T>::updateCache(int32_t ar_len, int32_t n_past, int32_t n_update, const std::vector<bool>& selected) {
  if (cur_ar_len_ != ar_len) {
    MLLM_ERROR("Current AR length ({}) is not matched with target AR length ({}). Please rearrange cache first.", cur_ar_len_,
               ar_len);
    exit(1);
  }
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    updateKey(k_cache_[layer], n_past, n_update, selected);
    updateValue(v_cache_[layer], n_past, n_update, selected);
  }
}

template<typename T>
void KVCacheManager<T>::updateKey(KVCache<T>& k_cache, int32_t n_past, int32_t n_update, const std::vector<bool>& selected) {
  T* write_ptr = k_cache.buffer;
  T* read_ptr = k_cache.output_buffer;
  const int32_t copy_size = n_update * sizeof(T);
  const int32_t iter_size = (cur_ar_len_ == config_.context_len) ? config_.context_len : config_.context_len - cur_ar_len_;
  const int32_t out_size = cur_ar_len_;
  const int32_t past_size = n_past;
  const int32_t n_iter = config_.head_dim * config_.num_heads;

  write_ptr += past_size;
  if (selected.empty()) {
    for (int i = 0; i < n_iter; ++i) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    std::vector<int32_t> true_indices(n_update);
    for (int i = 0, j = 0; i < selected.size() && j < n_update; ++i) {
      if (selected[i]) { true_indices[j++] = i; }
    }
    for (int i = 0; i < n_iter; ++i) {
      for (int j = 0; j < n_update; ++j) { write_ptr[j] = read_ptr[true_indices[j]]; }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

template<typename T>
void KVCacheManager<T>::updateValue(KVCache<T>& v_cache, int32_t n_past, int32_t n_update, const std::vector<bool>& selected) {
  T* write_ptr = v_cache.buffer;
  T* read_ptr = v_cache.output_buffer;
  const int32_t copy_size = n_update * config_.head_dim * sizeof(T);
  const int32_t past_size = n_past * config_.head_dim;
  const int32_t n_iter = config_.num_heads;
  const int32_t iter_size = (cur_ar_len_ == config_.context_len) ? config_.context_len * config_.head_dim
                                                                 : (config_.context_len - cur_ar_len_) * config_.head_dim;
  const int32_t out_size = cur_ar_len_ * config_.head_dim;

  write_ptr += past_size;

  if (selected.empty()) {
    for (int i = 0; i < n_iter; i++) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    for (int i = 0; i < n_iter; i++) {
      auto wp = write_ptr;
      auto rp = read_ptr;
      int32_t update_cnt = 0;
      for (auto sel : selected) {
        if (sel) {
          std::memcpy(wp, rp, config_.head_dim * sizeof(T));
          wp += config_.head_dim;
          update_cnt++;
        }
        rp += config_.head_dim;
        if (update_cnt == n_update) break;
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

// Explicit instantiations
template class KVCacheManager<uint16_t>;
template class KVCacheManager<uint8_t>;

}  // namespace mllm::qnn::aot
