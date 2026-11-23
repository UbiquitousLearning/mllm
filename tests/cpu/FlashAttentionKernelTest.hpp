// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1/fwd_bshd.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"

#include "KernelTestHelper.hpp"

using namespace mllm;  // NOLINT

namespace fa2_kernel_test {

inline void testIdentityRescale(int batch, int q_heads, int kv_heads, int q_len, int kv_len, int dim) {
  using mllm::cpu::flash_attn2::fwd_bshd;
  using mllm::cpu::flash_attn2::details::__ArmArchTag;
  const auto q_size = static_cast<size_t>(batch) * q_len * q_heads * dim;
  const auto kv_size = static_cast<size_t>(batch) * kv_len * kv_heads * dim;
  std::vector<float> q(q_size), k(kv_size), v(kv_size), legacy(q_size), optimized(q_size);
  for (size_t i = 0; i < q_size; ++i) q[i] = static_cast<float>(static_cast<int>((i * 11) % 23) - 11) / 16.f;
  for (size_t i = 0; i < kv_size; ++i) {
    // Consecutive equal keys exercise the identity transition; changing
    // their magnitude and direction also exercises the rescale path.
    const size_t row_size = static_cast<size_t>(kv_heads) * dim;
    const size_t pair = (i / row_size) / 2;
    const int key = static_cast<int>(((i % row_size) * 7 + pair * 3) % 31) - 15;
    k[i] = static_cast<float>(key) / 8.f;
    v[i] = static_cast<float>(static_cast<int>((i * 17) % 29) - 14) / 32.f;
  }
  fwd_bshd<__ArmArchTag, float, float, float, float, float, true, false>(batch, q_heads, kv_heads, q_len, kv_len, dim, q.data(),
                                                                         k.data(), v.data(), legacy.data(), 1);
  fwd_bshd<__ArmArchTag, float, float, float, float, float, true, true>(batch, q_heads, kv_heads, q_len, kv_len, dim, q.data(),
                                                                        k.data(), v.data(), optimized.data(), 1);
  for (size_t i = 0; i < q_size; ++i) {
    ASSERT_TRUE(std::isfinite(legacy[i]));
    ASSERT_EQ(std::bit_cast<uint32_t>(legacy[i]), std::bit_cast<uint32_t>(optimized[i])) << "element " << i;
  }
}

}  // namespace fa2_kernel_test

class FlashAttn2Module : public nn::Module {
 public:
  FlashAttn2Module() = default;

  FlashAttn2Module(int H_Q, int H_KV) : nn::Module() {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // inputs is Q, K_indices, V_indices
    return {nn::functional::flashAttention2(inputs[0], inputs[1], inputs[2])};
  }
};

class FA2EagerModule : public nn::Module {
 public:
  FA2EagerModule() : nn::Module() {}

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

class FlashAttn2KernelTest : public KernelTest {
 public:
  FlashAttn2KernelTest() = default;
  ~FlashAttn2KernelTest() override = default;

  bool testRadixAttnOnce(const std::unordered_map<std::string, int32_t>& cfg) {
    int B = 1;
    int H_Q = cfg.at("H_Q");
    int H_KV = cfg.at("H_KV");
    int S_Q = cfg.at("S_Q");
    int S_KV = cfg.at("S_KV");
    int D = cfg.at("D");

    FA2EagerModule eager_attn;
    FlashAttn2Module flash_attn2(H_Q, H_KV);

    // Create Q, K, V
    auto Q = Tensor::random({B, S_Q, H_Q, D}, -10.f, 10.f);
    auto K = Tensor::random({B, S_KV, H_KV, D}, -10.f, 10.f);
    auto V = Tensor::random({B, S_KV, H_KV, D}, -10.f, 10.f);

    // Compute eager
    Tensor gt = eager_attn(Q, K, V)[0];
    Tensor predict = flash_attn2(Q, K, V)[0];

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
