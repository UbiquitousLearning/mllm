// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include "mllm/core/Storage.hpp"
#include "mllm/backends/base/Allocator.hpp"

namespace mllm::qnn::aot {

template<typename T>
struct KVCache {
  std::shared_ptr<mllm::Storage> buffer_storage;
  std::shared_ptr<mllm::Storage> output_buffer_storage;
  T* buffer;
  T* output_buffer;
};

struct KVCacheConfig {
  int32_t context_len;
  int64_t head_dim;
  int32_t max_ar_len;
  int32_t max_cache_len;
  int64_t num_heads;
  int64_t num_layers;
};

template<typename T>
class KVCacheManager {
 public:
  explicit KVCacheManager(KVCacheConfig config);
  ~KVCacheManager() = default;

  void initCache(mllm::Allocator* allocator, int32_t ar_len);
  void rearrangeCache(int32_t ar_len_dst);

  void initAttentionMask(uint16_t* attention_mask, const std::vector<int32_t>& attention_map, int32_t ar_len, int32_t n_past);

  void initAttentionMask(uint16_t* attention_mask, const std::vector<int32_t>& attention_map, int32_t ar_len, int32_t n_past,
                         int32_t sliding_window, const std::vector<int32_t>& position_offset = {});

  void updateAttentionMask(uint16_t* attention_mask, int32_t ar_len, int32_t n_past, int32_t n_update);

  void updateAttentionMask(uint16_t* attention_mask, int32_t ar_len, int32_t n_past, int32_t n_update, int32_t sliding_window,
                           const std::vector<int32_t>& position_offset = {});

  void updateCache(int32_t ar_len, int32_t n_past, int32_t n_update, const std::vector<bool>& selected);

  const std::vector<KVCache<T>>& getKCache() const { return k_cache_; }
  const std::vector<KVCache<T>>& getVCache() const { return v_cache_; }
  [[nodiscard]] size_t getTotalCacheSizeInBytes() const { return total_cache_size_; }

 private:
  void rearrangeKey(KVCache<T>& k_cache, int32_t ar_len_dst);
  void rearrangeValue(KVCache<T>& v_cache, int32_t ar_len_dst);
  void updateKey(KVCache<T>& k_cache, int32_t n_past, int32_t n_update, const std::vector<bool>& selected);
  void updateValue(KVCache<T>& v_cache, int32_t n_past, int32_t n_update, const std::vector<bool>& selected);

  KVCacheConfig config_;
  size_t total_cache_size_;
  int32_t cur_ar_len_;
  std::vector<KVCache<T>> k_cache_;
  std::vector<KVCache<T>> v_cache_;
};

}  // namespace mllm::qnn::aot