// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>
#include <ranges>
#include <fstream>
#include "mllm/utils/Common.hpp"
#include "mllm/nn/lmcache/PrefixCache.hpp"
#include "mllm/engine/prefix_cache/TLB.hpp"
#include "mllm/engine/prefix_cache/Allocator.hpp"

namespace mllm::nn {

PrefixCache::PrefixCache(const PrefixCacheOptions& options) : options_(options) {}

void PrefixCache::dot(const std::string& fp) const { std::ofstream(fp) << tree_->dot(); }

PrefixCache::~PrefixCache() = default;

void PrefixCache::promote(const std::vector<int64_t>& token_ids,
                          const std::vector<std::vector<prefix_cache::vp_addr_t>>& key_cache_addresses,
                          const std::vector<std::vector<prefix_cache::vp_addr_t>>& value_cache_addresses, int64_t extra_key) {}

prefix_cache::vp_addr_t PrefixCache::allocKey(int layer_idx) { return -1; }

prefix_cache::vp_addr_t PrefixCache::allocValue(int layer_idx) { return -1; }

void PrefixCache::freeKey(int layer_idx, prefix_cache::vp_addr_t addr) {}

void PrefixCache::freeValue(int layer_idx, prefix_cache::vp_addr_t addr) {}

char* PrefixCache::physicalAddrKey(int layer_idx, prefix_cache::vp_addr_t addr) { return nullptr; }

char* PrefixCache::physicalAddrValue(int layer_idx, prefix_cache::vp_addr_t addr) { return nullptr; }

void PrefixCache::_initFullAttention() {}

void PrefixCache::_initSlidingWindowAttention() {}

void PrefixCache::_validateKeyTokenShape(Tensor& key) {}

void PrefixCache::_validateValueTokenShape(Tensor& value) {}

void PrefixCache::prefetchKey(int layer_idx, prefix_cache::vp_addr_t addr) {}

void PrefixCache::prefetchValue(int layer_idx, prefix_cache::vp_addr_t addr) {}

void PrefixCache::purgeKey(int layer_idx, prefix_cache::vp_addr_t addr) {}

void PrefixCache::purgeValue(int layer_idx, prefix_cache::vp_addr_t addr) {}

prefix_cache::RadixSearchResult PrefixCache::find(const std::vector<int64_t>& token_ids) {
  return tree_->search(prefix_cache::RadixTreeNodeKey(prefix_cache::VectorView<int64_t>(token_ids)));
}

CpuPrefixCache::CpuPrefixCache(const PrefixCacheOptions& options) : PrefixCache(options) {
  if (options.hybrid_sliding_window_cache.value_or(false)) {
    _initSlidingWindowAttention();
  } else {
    _initFullAttention();
  }
  prefix_cache::RadixTreeOptions radix_tree_options = {.enable_lru_eviction = false,      // Not support yet
                                                       .eviction_threshold = 0.9f,        // Not support yet
                                                       .enable_path_compression = false,  // Not support yet
                                                       .min_compression_length = 2,       // Not support yet
                                                       .transformer_blocks_num = options_.total_layers};
  tree_ = std::make_shared<prefix_cache::RadixTree>(radix_tree_options);
}

prefix_cache::vp_addr_t CpuPrefixCache::allocKey(int layer_idx) { return caches_[layer_idx].first->alloc(); }

prefix_cache::vp_addr_t CpuPrefixCache::allocValue(int layer_idx) { return caches_[layer_idx].second->alloc(); }

void CpuPrefixCache::promote(const std::vector<int64_t>& token_ids,
                             const std::vector<std::vector<prefix_cache::vp_addr_t>>& key_cache_addresses,
                             const std::vector<std::vector<prefix_cache::vp_addr_t>>& value_cache_addresses,
                             int64_t extra_key) {
  prefix_cache::RadixTreeNodeValue value;
  for (int i = 0; i < options_.total_layers; ++i) {
    value.k_cache_addresses.emplace_back(key_cache_addresses[i]);
    value.v_cache_addresses.emplace_back(value_cache_addresses[i]);
  }
  tree_->insert(prefix_cache::RadixTreeNodeKey(prefix_cache::VectorView<int64_t>(token_ids), extra_key), value);
}

void CpuPrefixCache::evictSlidingWindowLayerOn(const std::vector<int64_t>& token_ids) {
  // Check the sliding window layers if they need to be evicted
  // Remember that the token_ids is full and not partial
  if (!options_.hybrid_sliding_window_cache.value_or(false)) { return; }
  if (token_ids.size() <= options_.sliding_window_size.value()) { return; }

  // We evict the tokens outof sliding window.
  // 1. find token_ids path
  auto result = find(token_ids);

  for (int layer_idx = 0; layer_idx < options_.total_layers; ++layer_idx) {
    // Only perform on sliding window
    if (options_.sliding_window_layers.value()[layer_idx]) {
      // loop on path to delete tokens
      if (result.success && result.matched_length > options_.sliding_window_size.value()) {
        // We need to evict
        int length_cnt = 0;

        // Loop from back and matching
        for (auto& it : std::ranges::reverse_view(result.path)) {
          auto& radix_tree_node = it.first->value;
          auto this_node_matched_length = it.second;

          // We need to cut down those tokens outof sliding window
          if (length_cnt + this_node_matched_length > options_.sliding_window_size.value()) {
            int tokens_to_evict = std::min(this_node_matched_length,
                                           length_cnt + this_node_matched_length - options_.sliding_window_size.value());
            for (int idx = 0; idx < tokens_to_evict; ++idx) {
              freeKey(layer_idx, radix_tree_node.k_cache_addresses[layer_idx][idx]);
              freeValue(layer_idx, radix_tree_node.v_cache_addresses[layer_idx][idx]);
              radix_tree_node.k_cache_addresses[layer_idx][idx] = INVALID_VP_ADDR;
              radix_tree_node.v_cache_addresses[layer_idx][idx] = INVALID_VP_ADDR;
            }
          }

          // Update length_cnt
          length_cnt += this_node_matched_length;
        }
      }
    }
  }
}

void CpuPrefixCache::freeKey(int layer_idx, prefix_cache::vp_addr_t addr) {
  if (addr == INVALID_VP_ADDR) return;
  caches_[layer_idx].first->free(addr);
}

void CpuPrefixCache::freeValue(int layer_idx, prefix_cache::vp_addr_t addr) {
  if (addr == INVALID_VP_ADDR) return;
  caches_[layer_idx].second->free(addr);
}

char* CpuPrefixCache::physicalAddrKey(int layer_idx, prefix_cache::vp_addr_t addr) {
  return caches_[layer_idx].first->physicalAddr(addr);
}

char* CpuPrefixCache::physicalAddrValue(int layer_idx, prefix_cache::vp_addr_t addr) {
  return caches_[layer_idx].second->physicalAddr(addr);
}

void CpuPrefixCache::prefetchKey(int layer_idx, prefix_cache::vp_addr_t addr) { caches_[layer_idx].first->prefetch(addr); }

void CpuPrefixCache::prefetchValue(int layer_idx, prefix_cache::vp_addr_t addr) { caches_[layer_idx].second->prefetch(addr); }

void CpuPrefixCache::purgeKey(int layer_idx, prefix_cache::vp_addr_t addr) { caches_[layer_idx].first->purge(addr); }

void CpuPrefixCache::purgeValue(int layer_idx, prefix_cache::vp_addr_t addr) { caches_[layer_idx].second->purge(addr); }

void CpuPrefixCache::_initFullAttention() {
  caches_.resize(options_.total_layers);
  for (int layer_idx = 0; layer_idx < options_.total_layers; ++layer_idx) {
    auto opt_1 =
        prefix_cache::PrefixCacheAllocatorOptions{.per_k_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                  .per_v_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                  .k_dtype = options_.k_dtype,
                                                  .v_dtype = options_.v_dtype,

                                                  // CUDA things
                                                  .enable_cuda = options_.enable_cuda,
                                                  .cuda_mem_base = options_.cuda_mem_base,

                                                  // cpu things
                                                  .enable_cpu_hierarchy_memory = true,
                                                  .zen_fs_options = {
                                                      .record = false,
                                                      .working_dir = options_.zen_fs_working_dir,
                                                      .blob_bits_size = options_.zen_fs_blob_bits_size,
                                                      .page_bits = options_.zen_fs_page_bits,
                                                      .lane_bits = options_.zen_fs_lane_bits,
                                                      .per_k_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                      .per_v_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                      .k_dtype = options_.k_dtype,
                                                      .v_dtype = options_.v_dtype,
                                                      .mmap_type = options_.zen_fs_mem_type,
                                                  }};

    auto opt_2 =
        prefix_cache::PrefixCacheAllocatorOptions{.per_k_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                  .per_v_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                  .k_dtype = options_.k_dtype,
                                                  .v_dtype = options_.v_dtype,

                                                  // CUDA things
                                                  .enable_cuda = options_.enable_cuda,
                                                  .cuda_mem_base = options_.cuda_mem_base,

                                                  // cpu things
                                                  .enable_cpu_hierarchy_memory = true,
                                                  .zen_fs_options = {
                                                      .record = false,
                                                      .working_dir = options_.zen_fs_working_dir,
                                                      .blob_bits_size = options_.zen_fs_blob_bits_size,
                                                      .page_bits = options_.zen_fs_page_bits,
                                                      .lane_bits = options_.zen_fs_lane_bits,
                                                      .per_k_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                      .per_v_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                      .k_dtype = options_.k_dtype,
                                                      .v_dtype = options_.v_dtype,
                                                      .mmap_type = options_.zen_fs_mem_type,
                                                  }};
    // Key and Value
    caches_[layer_idx] = {std::make_shared<prefix_cache::_HiCPUAllocator>(opt_1.zen_fs_options),
                          std::make_shared<prefix_cache::_HiCPUAllocator>(opt_2.zen_fs_options)};
  }  // namespace mllm::nn
}

void CpuPrefixCache::_initSlidingWindowAttention() {
  MLLM_RT_ASSERT(options_.sliding_window_layers.has_value());

  // We calculate how many tokens we actually need.
  size_t sliding_window_lane_page_sum_size = std::log2(options_.sliding_window_size.value());

  caches_.resize(options_.total_layers);
  for (int layer_idx = 0; layer_idx < options_.total_layers; ++layer_idx) {
    if (options_.sliding_window_layers.value()[layer_idx]) {
      auto opt_1 =
          prefix_cache::PrefixCacheAllocatorOptions{.per_k_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                    .per_v_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                    .k_dtype = options_.k_dtype,
                                                    .v_dtype = options_.v_dtype,

                                                    // CUDA things
                                                    .enable_cuda = options_.enable_cuda,
                                                    .cuda_mem_base = options_.cuda_mem_base,

                                                    // cpu things
                                                    .enable_cpu_hierarchy_memory = true,
                                                    .zen_fs_options = {
                                                        .record = false,
                                                        .working_dir = options_.zen_fs_working_dir,
                                                        .blob_bits_size = 32 - sliding_window_lane_page_sum_size,
                                                        .page_bits = 0,
                                                        .lane_bits = sliding_window_lane_page_sum_size,
                                                        .per_k_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                        .per_v_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                        .k_dtype = options_.k_dtype,
                                                        .v_dtype = options_.v_dtype,
                                                        .mmap_type = options_.zen_fs_mem_type,
                                                    }};
      auto opt_2 =
          prefix_cache::PrefixCacheAllocatorOptions{.per_k_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                    .per_v_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                    .k_dtype = options_.k_dtype,
                                                    .v_dtype = options_.v_dtype,

                                                    // CUDA things
                                                    .enable_cuda = options_.enable_cuda,
                                                    .cuda_mem_base = options_.cuda_mem_base,

                                                    // cpu things
                                                    .enable_cpu_hierarchy_memory = true,
                                                    .zen_fs_options = {
                                                        .record = false,
                                                        .working_dir = options_.zen_fs_working_dir,
                                                        .blob_bits_size = 32 - sliding_window_lane_page_sum_size,
                                                        .page_bits = 0,
                                                        .lane_bits = sliding_window_lane_page_sum_size,
                                                        .per_k_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                        .per_v_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                        .k_dtype = options_.k_dtype,
                                                        .v_dtype = options_.v_dtype,
                                                        .mmap_type = options_.zen_fs_mem_type,
                                                    }};
      caches_[layer_idx] = {std::make_shared<prefix_cache::_HiCPUAllocator>(opt_1.zen_fs_options),
                            std::make_shared<prefix_cache::_HiCPUAllocator>(opt_2.zen_fs_options)};
    } else {
      auto opt_1 =
          prefix_cache::PrefixCacheAllocatorOptions{.per_k_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                    .per_v_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                    .k_dtype = options_.k_dtype,
                                                    .v_dtype = options_.v_dtype,

                                                    // CUDA things
                                                    .enable_cuda = options_.enable_cuda,
                                                    .cuda_mem_base = options_.cuda_mem_base,

                                                    // cpu things
                                                    .enable_cpu_hierarchy_memory = true,
                                                    .zen_fs_options = {
                                                        .record = false,
                                                        .working_dir = options_.zen_fs_working_dir,
                                                        .blob_bits_size = options_.zen_fs_blob_bits_size,
                                                        .page_bits = options_.zen_fs_page_bits,
                                                        .lane_bits = options_.zen_fs_lane_bits,
                                                        .per_k_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                        .per_v_token_ele = static_cast<size_t>(options_.per_k_token_ele_num),
                                                        .k_dtype = options_.k_dtype,
                                                        .v_dtype = options_.v_dtype,
                                                        .mmap_type = options_.zen_fs_mem_type,
                                                    }};
      auto opt_2 =
          prefix_cache::PrefixCacheAllocatorOptions{.per_k_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                    .per_v_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                    .k_dtype = options_.k_dtype,
                                                    .v_dtype = options_.v_dtype,

                                                    // CUDA things
                                                    .enable_cuda = options_.enable_cuda,
                                                    .cuda_mem_base = options_.cuda_mem_base,

                                                    // cpu things
                                                    .enable_cpu_hierarchy_memory = true,
                                                    .zen_fs_options = {
                                                        .record = false,
                                                        .working_dir = options_.zen_fs_working_dir,
                                                        .blob_bits_size = options_.zen_fs_blob_bits_size,
                                                        .page_bits = options_.zen_fs_page_bits,
                                                        .lane_bits = options_.zen_fs_lane_bits,
                                                        .per_k_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                        .per_v_token_ele = static_cast<size_t>(options_.per_v_token_ele_num),
                                                        .k_dtype = options_.k_dtype,
                                                        .v_dtype = options_.v_dtype,
                                                        .mmap_type = options_.zen_fs_mem_type,
                                                    }};
      caches_[layer_idx] = {std::make_shared<prefix_cache::_HiCPUAllocator>(opt_1.zen_fs_options),
                            std::make_shared<prefix_cache::_HiCPUAllocator>(opt_2.zen_fs_options)};
    }
  }  // namespace mllm::nn
}

void CpuPrefixCache::_validateKeyTokenShape(Tensor& key) {
  MLLM_RT_ASSERT_EQ(key.numel(), options_.per_k_token_ele_num);
  MLLM_RT_ASSERT_EQ(key.device(), kCPU);
}

void CpuPrefixCache::_validateValueTokenShape(Tensor& value) {
  MLLM_RT_ASSERT_EQ(value.numel(), options_.per_v_token_ele_num);
  MLLM_RT_ASSERT_EQ(value.device(), kCPU);
}

}  // namespace mllm::nn
