// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/utils/Common.hpp"

using namespace mllm;  // NOLINT

int main() {
  mllm::initializeContext();

  constexpr int max_cache_length = 6;
  constexpr int layer_nums = 12;
  constexpr int q_heads = 4;
  constexpr int kv_heads = 2;
  constexpr int kv_dim = 6;

  nn::StaticCache cache(max_cache_length, layer_nums, q_heads, kv_heads, kv_dim, kFloat32, kFloat32, kCPU, false);

  for (int i = 0; i < layer_nums; ++i) { MLLM_RT_ASSERT_EQ(cache.getCurrentSeqCnt(i), 0); }

  mllm::shutdownContext();
}
