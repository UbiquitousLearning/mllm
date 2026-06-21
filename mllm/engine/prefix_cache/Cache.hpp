// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/prefix_cache/TLB.hpp"
#include "mllm/engine/prefix_cache/Allocator.hpp"
#include "mllm/engine/prefix_cache/RadixTree.hpp"

namespace mllm::prefix_cache {

struct CacheOptions {
  RadixTreeOptions radix_tree_options = {.enable_lru_eviction = false,
                                         .eviction_threshold = 0.9f,
                                         .enable_path_compression = false,
                                         .min_compression_length = 2,
                                         .transformer_blocks_num = 1};
  PrefixCacheAllocatorOptions allocator_options;
};

class Cache {
 public:
  Cache() = default;

  explicit Cache(const CacheOptions& options);

  void promote(const std::vector<int64_t>& token_ids, const std::vector<std::vector<vp_addr_t>>& key_cache_addresses,
               const std::vector<std::vector<vp_addr_t>>& value_cache_addresses, int64_t extra_key = 0);

  RadixSearchResult find(const std::vector<int64_t>& token_ids);

  // Low level APIs
  vp_addr_t alloc(DeviceTypes device_type);

  void free(DeviceTypes device_type, vp_addr_t addr);

  char* physicalAddr(vp_addr_t addr);

  void prefetch(DeviceTypes device_type, vp_addr_t addr);

  void purge(DeviceTypes device_type, vp_addr_t addr);

  void dot(const std::string& fp) const;

 private:
  CacheOptions options_;

  RadixTree tree_;
  PrefixCacheAllocator allocator_;
};

}  // namespace mllm::prefix_cache
