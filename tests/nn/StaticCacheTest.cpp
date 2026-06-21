// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/utils/Common.hpp"

using namespace mllm;  // NOLINT

int main() {
  mllm::initializeContext();

  constexpr int max_cache_length = 32;
  constexpr int layer_nums = 12;
  constexpr int q_heads = 4;
  constexpr int kv_heads = 2;
  constexpr int kv_dim = 6;

  nn::StaticCache cache(max_cache_length, layer_nums, q_heads, kv_heads, kv_dim, kFloat32, kFloat32, kCPU, false);

  for (int i = 0; i < layer_nums; ++i) { MLLM_RT_ASSERT_EQ(cache.getCurrentSeqCnt(i), 0); }

  // 1. Insert KV, [B, H, S=1, D]
  {
    print("Insert KV, [B, H, S=1, D]");
    auto k = Tensor::random({1, kv_heads, 1, kv_dim});
    auto v = Tensor::random({1, kv_heads, 1, kv_dim});

    auto [key, value] = cache.updateKVCache(0, k, v);

    print(key.shape(), value.shape());
    print(key);
    print(value);
  }

  // 1. Insert KV, [B, H, S=2, D]
  {
    print("Insert KV, [B, H, S=2, D]");
    auto k = Tensor::random({1, kv_heads, 2, kv_dim});
    auto v = Tensor::random({1, kv_heads, 2, kv_dim});

    auto [key, value] = cache.updateKVCache(0, k, v);

    print(key.shape(), value.shape());
    print(key);
    print(value);
  }

  mllm::shutdownContext();
}
