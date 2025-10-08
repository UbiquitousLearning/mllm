// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <limits>
#include <unordered_map>

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/PrefixCache.hpp"

#include "KernelTestHelper.hpp"

using namespace mllm;  // NOLINT

class RadixAttnModule : public nn::Module {
  nn::RadixAttn attn_;

 public:
  RadixAttnModule() = default;

  RadixAttnModule(int H_Q, int H_KV) : nn::Module() { attn_ = reg<nn::RadixAttn>("attn", H_Q, H_KV); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // inputs is Q, K_indices, V_indices
    return {attn_(inputs[0], inputs[1], inputs[2])};
  }
};

class EagerModule : public nn::Module {
 public:
  EagerModule() : nn::Module() {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // inputs is Q, K_indices, V_indices
    // Q, K, V is [B, S, H, D]
    auto Q = inputs[0];
    auto K = inputs[1];
    auto V = inputs[2];

    auto h_q = Q.shape()[2];
    auto h_kv = K.shape()[2];
    auto head_dim = Q.shape()[3];

    // Q, K, V is [B, H, S, D]
    Q = Q.transpose(1, 2);
    K = K.transpose(1, 2).repeat(h_q / h_kv, 1);
    V = V.transpose(1, 2).repeat(h_q / h_kv, 1);

    // Attention Weight
    // [B, H, S, S]
    auto attn = nn::functional::matmul(Q, K, false, true) * (1.f / sqrtf(head_dim));

    // Make mask
    auto S_Q = Q.shape()[2];
    auto S_KV = K.shape()[2];
    auto mask = Tensor::zeros({1, 1, S_Q, S_KV});
    {
      auto ptr = mask.ptr<float>();
      int __delta = S_KV - S_Q;
      for (int s_q_idx = 0; s_q_idx < S_Q; s_q_idx++) {
        int S_KV_BOUND = std::min(__delta + s_q_idx + 1, S_KV);
        for (int s_kv_idx = S_KV_BOUND; s_kv_idx < S_KV; s_kv_idx++) {
          ptr[s_q_idx * S_KV + s_kv_idx] = -std::numeric_limits<float>::infinity();
        }
      }
    }

    attn = nn::functional::softmax(attn + mask, -1);
    // [B, H, S, D]
    auto output = nn::functional::matmul(attn, V);
    // [B, S, H, D]
    output = output.transpose(1, 2);

    return {output};
  }
};

class RadixAttnKernelTest : public KernelTest {
 public:
  RadixAttnKernelTest() = default;
  ~RadixAttnKernelTest() override = default;

  bool testRadixAttnOnce(const std::unordered_map<std::string, int32_t>& cfg) {
    int B = 1;
    int H_Q = cfg.at("H_Q");
    int H_KV = cfg.at("H_KV");
    int S_Q = cfg.at("S_Q");
    int S_KV = cfg.at("S_KV");
    int D = cfg.at("D");

    mllm::prefix_cache::CacheOptions opt{
        .radix_tree_options = {.enable_lru_eviction = false,
                               .eviction_threshold = 0.9f,
                               .enable_path_compression = false,
                               .min_compression_length = 2,
                               .transformer_blocks_num = 1},
        .allocator_options = {// Normal things.
                              .per_k_token_ele = static_cast<size_t>(H_KV * D),
                              .per_v_token_ele = static_cast<size_t>(H_KV * D),
                              .k_dtype = mllm::kFloat32,
                              .v_dtype = mllm::kFloat32,

                              // CUDA things.
                              .enable_cuda = false,
                              .cuda_mem_base = 0x100000,

                              // CPU things.
                              .enable_cpu_hierarchy_memory = true,
                              .zen_fs_options = {
                                  .record = false,
                                  .working_dir = ".",
                                  .blob_bits_size = 20,
                                  .page_bits = 7,
                                  .lane_bits = 5,
                                  .per_k_token_ele = static_cast<size_t>(H_KV * D),
                                  .per_v_token_ele = static_cast<size_t>(H_KV * D),
                                  .k_dtype = mllm::kFloat32,
                                  .v_dtype = mllm::kFloat32,
                                  .mmap_type = mllm::prefix_cache::ZenFSBlobMMapType::kAnonymous,
                              }}};
    EagerModule eager_attn;
    RadixAttnModule radix_attn(H_Q, H_KV);
    prefix_cache::Cache cache(opt);

    // Create Q, K, V
    auto Q = Tensor::random({B, S_Q, H_Q, D}, -10.f, 10.f);
    auto K = Tensor::random({B, S_KV, H_KV, D}, -10.f, 10.f);
    auto V = Tensor::random({B, S_KV, H_KV, D}, -10.f, 10.f);

    // Insert K and V into Cache and Radix Tree
    std::vector<prefix_cache::vp_addr_t> k_cache_addrs;
    std::vector<prefix_cache::vp_addr_t> v_cache_addrs;
    for (int i = 0; i < S_KV; i++) {
      k_cache_addrs.push_back(cache.alloc(kCPU));
      v_cache_addrs.push_back(cache.alloc(kCPU));
    }
    std::vector<char*> k_cache_ptrs;
    std::vector<char*> v_cache_ptrs;
    for (int i = 0; i < S_KV; i++) {
      k_cache_ptrs.push_back(cache.physicalAddr(k_cache_addrs[i]));
      v_cache_ptrs.push_back(cache.physicalAddr(v_cache_addrs[i]));
    }
    auto k_cache_indices = Tensor::refVectorData(k_cache_ptrs, {S_KV}, kInt64);
    auto v_cache_indices = Tensor::refVectorData(v_cache_ptrs, {S_KV}, kInt64);

    nn::functional::scatter2Shards(K, k_cache_indices, 1);
    nn::functional::scatter2Shards(V, v_cache_indices, 1);

    // loop check if shards is correct
    for (int i = 0; i < S_KV; ++i) {
      auto prd_ptr = (float*)k_cache_ptrs[i];
      auto gt_ptr = K.offsettedPtr<float>({0, i, 0, 0});
      for (int j = 0; j < H_KV * D; ++j) {
        if (prd_ptr[j] != gt_ptr[j]) {
          print("Error at: ", i, j);
          print("prd: ", prd_ptr[j], " gt: ", gt_ptr[j]);
          return false;
        }
      }
    }

    // Compute eager
    Tensor gt = eager_attn(Q, K, V)[0];
    Tensor predict = radix_attn(Q, k_cache_indices, v_cache_indices)[0];

    // Compare
    // rtol and atol set to 1e-2f is because:
    // 1. The eager softmax is approximate, but radix is not.
    auto result = test::allClose(gt, predict, 1e-2f, 1e-2f);
    if (!result) {
      print(result);
      print("S_Q and S_KV is", S_Q, S_KV);
      print(predict);
      return false;
    }
    return true;
  }

  bool testRadixAttn(const std::vector<std::unordered_map<std::string, int32_t>>& cfgs) {
    for (auto& cfg : cfgs) {
      if (!testRadixAttnOnce(cfg)) { return false; }
    }
    return true;
  }
};
