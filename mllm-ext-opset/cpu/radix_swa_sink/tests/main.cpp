#include <cmath>
#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/nn/lmcache/PrefixCache.hpp"
#include "mllm-ext-opset/cpu/radix_swa_sink/RadixAttnSwaSink.hpp"

mllm::Tensor radixAttnSWAwSink(const mllm::Tensor& Q, const mllm::Tensor& K_idx, const mllm::Tensor& V_idx,
                               const mllm::Tensor& s_aux, int batch, int q_head, int kv_head, int d_qk, int d_v,
                               int left_sliding_window, int seq_len) {
  auto& ctx = mllm::Context::instance();
  auto id = ctx.lookupCustomizedOpId(mllm::kCPU, "radix_attn_swa_sink");
  return ctx.buildOpAndSubmitTask((mllm::OpTypes)id,
                                  mllm::ext_opset::cpu::RadixAttnSwaSinkOptions{
                                      .B = batch,
                                      .q_head = q_head,
                                      .kv_head = kv_head,
                                      .D_QK = d_qk,
                                      .D_V = d_v,
                                      .cur_seq_len = seq_len,
                                      .sliding_window = left_sliding_window,
                                      .s_aux_enable = true,
                                  },
                                  {Q, K_idx, V_idx, s_aux})[0];
}

mllm::Tensor eagerSWAwSink(mllm::Tensor Q, mllm::Tensor K, mllm::Tensor V, mllm::Tensor s_aux,
                           const mllm::Tensor& causal_mask) {
  // Shape
  auto B = Q.size(0);
  auto S_Q = Q.size(1);
  auto H_Q = Q.size(2);
  auto S_KV = K.size(1);
  auto H_KV = K.size(2);
  auto D_QK = Q.size(-1);

  // repeat, BSHD
  auto key_states = K.repeat(H_Q / H_KV, 2);
  auto value_states = V.repeat(H_Q / H_KV, 2);

  // Transpose first. [B, S, H, D] -> [B, H, S, D]
  auto query_states = Q.transpose(1, 2);
  key_states = key_states.transpose(1, 2);
  value_states = value_states.transpose(1, 2);

  // Scale
  float scale = 1.f / std::sqrt(D_QK);

  // Get attn weight
  auto attn_weight = mllm::nn::functional::matmul(query_states, key_states, false, true) * scale;
  attn_weight = attn_weight + causal_mask;

  // [H] -> [B(1), H, S_Q, 1]
  auto sinks = s_aux.view({1, -1, 1, 1}).repeat(S_Q, 2);

  // [B, H, S_Q, S_KV]
  attn_weight = mllm::nn::functional::concat({attn_weight, sinks}, -1);
  attn_weight = mllm::nn::functional::softmax(attn_weight, -1);
  attn_weight = attn_weight[{mllm::kAll, mllm::kAll, mllm::kAll, {mllm::kAll, -1}}].contiguous();
  auto attn_output = mllm::nn::functional::matmul(attn_weight, value_states);

  // [B, H, S, D] -> [B, S, H, D]
  return attn_output.transpose(1, 2);
}

mllm::test::AllCloseResult testCase(int B, int S_Q, int S_KV, int H_Q, int H_KV, int D_QK, int D_V, int left_sliding_window,
                                    int cur_seq_len) {
  // CHECK
  MLLM_RT_ASSERT(S_KV <= left_sliding_window);

  // Create fake data
  auto s_aux = mllm::Tensor::random({H_Q}, -1.f, 1.f, mllm::kFloat32, mllm::kCPU);
  auto Q = mllm::Tensor::random({B, S_Q, H_Q, D_QK}, -1.f, 1.f, mllm::kFloat32, mllm::kCPU);
  auto K = mllm::Tensor::random({B, S_KV, H_KV, D_QK}, -1.f, 1.f, mllm::kFloat32, mllm::kCPU);
  auto V = mllm::Tensor::random({B, S_KV, H_KV, D_V}, -1.f, 1.f, mllm::kFloat32, mllm::kCPU);

  // Create Causal mask
  auto causal_mask = mllm::Tensor::zeros({1, 1, S_Q, std::min(left_sliding_window, S_KV)}, mllm::kFloat32, mllm::kCPU);
  {
    auto ptr = causal_mask.ptr<mllm::mllm_fp32_t>();
    for (int s_q_idx = 0; s_q_idx < S_Q; s_q_idx++) {
      // full attention start: s_q_idx + cur_seq_len - seq_q - window (include)
      // full attention end: s_q_idx + cur_seq_len - seq_q (include)

      auto full_attention_start = std::max(0, s_q_idx + cur_seq_len - S_Q - left_sliding_window);
      auto full_attention_end = std::min(cur_seq_len, s_q_idx + cur_seq_len - S_Q);

      // cur_window_size. S_KV may smaller then left_sliding_window
      auto cur_window_size = std::min(S_KV, left_sliding_window);

      // Compute should be done here.
      // for (int full_compute_idx = full_attention_start; full_compute_idx <= full_attention_end; ++full_compute_idx) {
      //   auto local_compute_idx = std::max(full_compute_idx, full_compute_idx - cur_window_size);
      // }

      // Loop on others
      for (int masked_idx = full_attention_end - full_attention_start + 1; masked_idx < cur_window_size; masked_idx++) {
        ptr[cur_window_size * s_q_idx + masked_idx] = -std::numeric_limits<float>::infinity();
      }
    }
  }

  // eager ref
  auto O_ref = eagerSWAwSink(Q, K, V, s_aux, causal_mask);

  mllm::nn::CpuPrefixCache cache(mllm::nn::PrefixCacheOptions{
      .max_batch_size = 1,
      .total_layers = 1,
      .per_k_token_ele_num = B * 1 * H_KV * D_QK,
      .per_v_token_ele_num = B * 1 * H_KV * D_V,
  });

  // Insert K and V into Cache and Radix Tree
  std::vector<mllm::prefix_cache::vp_addr_t> k_cache_addrs;
  std::vector<mllm::prefix_cache::vp_addr_t> v_cache_addrs;
  for (int i = 0; i < S_KV; i++) {
    k_cache_addrs.push_back(cache.allocKey(0));
    v_cache_addrs.push_back(cache.allocValue(0));
  }

  std::vector<char*> k_cache_ptrs;
  std::vector<char*> v_cache_ptrs;
  for (int i = 0; i < S_KV; i++) {
    k_cache_ptrs.push_back(cache.physicalAddrKey(0, k_cache_addrs[i]));
    v_cache_ptrs.push_back(cache.physicalAddrValue(0, v_cache_addrs[i]));
  }
  auto k_cache_indices = mllm::Tensor::refVectorData(k_cache_ptrs, {S_KV}, mllm::kInt64);
  auto v_cache_indices = mllm::Tensor::refVectorData(v_cache_ptrs, {S_KV}, mllm::kInt64);

  mllm::nn::functional::scatter2Shards(K, k_cache_indices, 1);
  mllm::nn::functional::scatter2Shards(V, v_cache_indices, 1);

  // eager
  auto O =
      radixAttnSWAwSink(Q, k_cache_indices, v_cache_indices, s_aux, 1, H_Q, H_KV, D_QK, D_V, left_sliding_window, cur_seq_len);

  auto ret = mllm::test::allClose(O, O_ref);

  if (!ret) {
    mllm::print(ret);
    mllm::print(O_ref);
    mllm::print(O);
  }

  return ret;
}

MLLM_MAIN({
  mllm::loadExtensionOpset("MllmExtOpSet.CPU.RadixAttnSwaSink");
  mllm::setRandomSeed(42);

  // inputs is B, S_Q, S_KV, H_Q, H_KV, D_QK, D_V, left_sliding_window, cur_seq_len

  // CASE: 1. For prefill S < left_sliding_window, should be normal mask
  MLLM_RT_ASSERT(testCase(1, 10, 10, 32, 4, 192, 128, 128, 10));

  // CASE: 2. For prefill S > left_sliding_window, kv cache is fixed to left_sliding_window rather then 256 due to sliding
  // window
  //
  // S_Q = 256
  // S_KV = 128(clamp to 128, original is 256, but we have window-size 128)
  // total-length = 256
  MLLM_RT_ASSERT(testCase(1, 256, std::min(128, 256), 32, 4, 192, 128, 128, 256));

  // CASE: 3. Decode, when cur_seq_len < left_sliding_window, which means S_KV == cur_seq_len
  MLLM_RT_ASSERT(testCase(1, 1, 32 + 1, 32, 4, 192, 128, 128, 32 + 1));

  // CASE: 4. Decode cur_seq_len > left_sliding_window, which means S_KV is always left_sliding_window
  MLLM_RT_ASSERT(testCase(1, 1, 128, 32, 4, 192, 128, 128, 5000 + 1));

  // CASE: 5. Append mode. cur_seq_len < left_sliding_window
  // S_Q: 10
  // S_KV: 20 + 10
  // window_size = 128
  // cur_seq_len = 20 + 10
  MLLM_RT_ASSERT(testCase(1, 10, 20 + 10, 32, 4, 192, 128, 128, 20 + 10));

  // CASE: 5. Append mode. cur_seq_len > left_sliding_window
  // S_Q: 10
  // S_KV: 128
  // window_size = 128
  // cur_seq_len = 128 + 10
  MLLM_RT_ASSERT(testCase(1, 10, 128, 32, 4, 192, 128, 128, 128 + 10));
})
