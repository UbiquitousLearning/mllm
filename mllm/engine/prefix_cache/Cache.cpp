// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <fstream>
#include "mllm/engine/prefix_cache/Cache.hpp"
#include "mllm/engine/prefix_cache/Allocator.hpp"
#include "mllm/engine/prefix_cache/RadixTree.hpp"

namespace mllm::prefix_cache {

Cache::Cache(const CacheOptions& options)
    : options_(options), tree_(options.radix_tree_options), allocator_(options.allocator_options) {}

void Cache::promote(const std::vector<int64_t>& token_ids, const std::vector<std::vector<vp_addr_t>>& key_cache_addresses,
                    const std::vector<std::vector<vp_addr_t>>& value_cache_addresses, int64_t extra_key) {
  RadixTreeNodeValue value;
  for (int i = 0; i < options_.radix_tree_options.transformer_blocks_num; ++i) {
    value.k_cache_addresses.emplace_back(key_cache_addresses[i]);
    value.v_cache_addresses.emplace_back(value_cache_addresses[i]);
  }
  tree_.insert(RadixTreeNodeKey(VectorView<int64_t>(token_ids), extra_key), value);
}

RadixSearchResult Cache::find(const std::vector<int64_t>& token_ids) {
  return tree_.search(RadixTreeNodeKey(VectorView<int64_t>(token_ids)));
}

// low level APIs
vp_addr_t Cache::alloc(DeviceTypes device_type) { return allocator_.alloc(device_type); }

void Cache::free(DeviceTypes device_type, vp_addr_t addr) { allocator_.free(device_type, addr); }

char* Cache::physicalAddr(vp_addr_t addr) { return allocator_.physicalAddr(addr); }

void Cache::prefetch(DeviceTypes device_type, vp_addr_t addr) { allocator_.prefetch(device_type, addr); }

void Cache::purge(DeviceTypes device_type, vp_addr_t addr) { allocator_.purge(device_type, addr); }

void Cache::dot(const std::string& fp) const { std::ofstream(fp) << tree_.dot(); }

}  // namespace mllm::prefix_cache
