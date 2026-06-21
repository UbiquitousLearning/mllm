#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/nn/lmcache/PrefixCache.hpp>
#include <mllm/engine/prefix_cache/TLB.hpp>

MLLM_MAIN({
  constexpr int B = 1;
  constexpr int H_KV = 4;
  constexpr int D_QK = 192;
  constexpr int D_V = 128;

  mllm::nn::CpuPrefixCache cache(mllm::nn::PrefixCacheOptions{
      .max_batch_size = 1,
      .total_layers = 1,
      .per_k_token_ele_num = B * 1 * H_KV * D_QK,
      .per_v_token_ele_num = B * 1 * H_KV * D_V,
      .hybrid_sliding_window_cache = true,
      .sliding_window_size = 8,
      .sliding_window_layers = std::vector<bool>{true},
  });

  std::vector<int64_t> token_ids;

  for (int i = 0; i < 16; ++i) {
    token_ids.push_back(i);

    auto result = cache.find(token_ids);
    std::vector<std::vector<mllm::prefix_cache::vp_addr_t>> k_cache_addresses = result.k_cache_addresses;
    std::vector<std::vector<mllm::prefix_cache::vp_addr_t>> v_cache_addresses = result.v_cache_addresses;

    // alloc a key and value
    auto k_va = cache.allocKey(0);
    auto v_va = cache.allocValue(0);
    k_cache_addresses[0].push_back(k_va);
    v_cache_addresses[0].push_back(v_va);

    MLLM_INFO("token: {}, key_va: {}, value_va: {}", i, k_va, v_va);

    // promote
    cache.promote(token_ids, k_cache_addresses, v_cache_addresses, 0);
    cache.evictSlidingWindowLayerOn(token_ids);
  }

  mllm::print(cache.find(token_ids).k_cache_addresses[0]);
  mllm::print(cache.find(token_ids).v_cache_addresses[0]);

  cache.dot("paged_attn_hybrid.dot");
})
