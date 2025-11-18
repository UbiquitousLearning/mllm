// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// This file implements prefix cache in paged attention. The page size is hard coded to 1. We use mllm/engine/prefix_cache to
// implement this class. Be aware that, things in mllm/engine/prefix_cache is low-level and not have fancy features such as
// sliding window attention supports or hybrid cache. We implement all fancy stuffs in this file.
//
// PrefixCache support features:
//
// 1. Full Paged Attention (all layers are paged attention)
// 2. Hybrid Cache: Paged Sliding window with Paged full attention.
//  NOTE: Hybrid Cache with sliding window attention not support radix cache.
//
// Diff with Cache in mllm/engine/prefix_cache:
// prefix_cache::Cache is more like a prototype that eval prefix_cache's accuracy. It's not contain high-level features, such as
// hybrid or compressed kv cache.
//

#pragma once

#include <memory>
#include <vector>
#include <optional>
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/prefix_cache/TLB.hpp"
#include "mllm/engine/prefix_cache/Cache.hpp"  // IWYU pragma: export

namespace mllm::nn {

struct PrefixCacheOptions {
  int max_batch_size = 1;

  // KV Token
  int total_layers = -1;
  int per_k_token_ele_num = -1;
  int per_v_token_ele_num = -1;
  DataTypes k_dtype = kFloat32;
  DataTypes v_dtype = kFloat32;

  // Algorithm based.
  // 1. Hybrid Sliding Window
  std::optional<bool> hybrid_sliding_window_cache = std::nullopt;
  std::optional<int> sliding_window_size = std::nullopt;
  std::optional<std::vector<bool>> sliding_window_layers = std::nullopt;

  // Defaults: User should change those settings in some situation
  std::string zen_fs_working_dir = ".";
  size_t zen_fs_blob_bits_size = 20;
  size_t zen_fs_page_bits = 7;
  size_t zen_fs_lane_bits = 5;
  prefix_cache::ZenFSBlobMMapType zen_fs_mem_type = prefix_cache::ZenFSBlobMMapType::kAnonymous;

  // Defaults: User have no will to modify this :)
  // CUDA things:
  bool enable_cuda = false;
  prefix_cache::vp_addr_t cuda_mem_base = 0x100000;
};

class PrefixCache {
 public:
  explicit PrefixCache(const PrefixCacheOptions& options);

  prefix_cache::RadixSearchResult find(const std::vector<int64_t>& token_ids);

  virtual ~PrefixCache();

  virtual void promote(const std::vector<int64_t>& token_ids,
                       const std::vector<std::vector<prefix_cache::vp_addr_t>>& key_cache_addresses,
                       const std::vector<std::vector<prefix_cache::vp_addr_t>>& value_cache_addresses, int64_t extra_key);

  virtual prefix_cache::vp_addr_t allocKey(int layer_idx);

  virtual prefix_cache::vp_addr_t allocValue(int layer_idx);

  virtual void freeKey(int layer_idx, prefix_cache::vp_addr_t addr);

  virtual void freeValue(int layer_idx, prefix_cache::vp_addr_t addr);

  virtual char* physicalAddrKey(int layer_idx, prefix_cache::vp_addr_t addr);

  virtual char* physicalAddrValue(int layer_idx, prefix_cache::vp_addr_t addr);

  virtual void _initFullAttention();

  virtual void _initSlidingWindowAttention();

  virtual void _validateKeyTokenShape(Tensor& key);

  virtual void _validateValueTokenShape(Tensor& value);

  virtual void prefetchKey(int layer_idx, prefix_cache::vp_addr_t addr);

  virtual void prefetchValue(int layer_idx, prefix_cache::vp_addr_t addr);

  virtual void purgeKey(int layer_idx, prefix_cache::vp_addr_t addr);

  virtual void purgeValue(int layer_idx, prefix_cache::vp_addr_t addr);

  void dot(const std::string& fp) const;

 protected:
  PrefixCacheOptions options_;
  std::shared_ptr<prefix_cache::RadixTree> tree_ = nullptr;
};

class CpuPrefixCache final : public PrefixCache {
 public:
  explicit CpuPrefixCache(const PrefixCacheOptions& options);

  void promote(const std::vector<int64_t>& token_ids,
               const std::vector<std::vector<prefix_cache::vp_addr_t>>& key_cache_addresses,
               const std::vector<std::vector<prefix_cache::vp_addr_t>>& value_cache_addresses, int64_t extra_key) override;

  /**
   * @brief This function will evict kv cache outside of sliding window layers
   *
   * @param token_ids
   */
  void evictSlidingWindowLayerOn(const std::vector<int64_t>& token_ids);

  prefix_cache::vp_addr_t allocKey(int layer_idx) override;

  prefix_cache::vp_addr_t allocValue(int layer_idx) override;

  void freeKey(int layer_idx, prefix_cache::vp_addr_t addr) override;

  void freeValue(int layer_idx, prefix_cache::vp_addr_t addr) override;

  char* physicalAddrKey(int layer_idx, prefix_cache::vp_addr_t addr) override;

  char* physicalAddrValue(int layer_idx, prefix_cache::vp_addr_t addr) override;

  void prefetchKey(int layer_idx, prefix_cache::vp_addr_t addr) override;

  void prefetchValue(int layer_idx, prefix_cache::vp_addr_t addr) override;

  void purgeKey(int layer_idx, prefix_cache::vp_addr_t addr) override;

  void purgeValue(int layer_idx, prefix_cache::vp_addr_t addr) override;

  void _initFullAttention() override;

  void _initSlidingWindowAttention() override;

  void _validateKeyTokenShape(Tensor& key) override;

  void _validateValueTokenShape(Tensor& value) override;

 private:
  // Each Layer each allocator for memory contiguous on CPU platform.
  std::vector<std::pair<prefix_cache::_AllocatorImpl::ptr_t, prefix_cache::_AllocatorImpl::ptr_t>> caches_;
};

}  // namespace mllm::nn
