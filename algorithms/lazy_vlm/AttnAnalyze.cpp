#include "AttnAnalyze.hpp"

#include <cmath>
#include <mllm/nn/Functional.hpp>

std::vector<int32_t> analyzePrefillAttn(mllm::Tensor attn, int32_t img_token_start, int32_t img_token_end,
                                        float pruning_ratio) {
  // Prefill attention inputs is [B=1, H, S_all, S_all]
  attn = attn.mean(1);     // [B=1, S_all, S_all]
  attn = attn.squeeze(0);  // [S_all, S_all]

  // [S_text, S_img]
  auto text_2_visual = attn[{{img_token_end, mllm::kAll}, {img_token_start, img_token_end}}].contiguous();
  text_2_visual = text_2_visual.sum(0);  // [S_img]

  // Calculate the topk tokens number
  MLLM_RT_ASSERT_EQ(text_2_visual.rank(), 1);
  auto S_img_tokens = text_2_visual.shape()[0];
  int32_t left_tokens = (int32_t)std::ceil(S_img_tokens * (1 - pruning_ratio));

  auto [_, indices] = mllm::nn::functional::topk(text_2_visual, left_tokens, /*dim*/ -1, /*largest*/ true, /*sorted*/ false);

  return indices.toVector<int32_t>();
}

std::vector<int32_t> analyzeDecodeAttn(mllm::Tensor attn, int32_t img_token_start, int32_t img_token_end, float pruning_ratio) {
  // Decode attention inputs is [B=1, H, 1, S_kv]
  // For decode phase, we have attention from current token to all previous tokens including visual tokens
  attn = attn.mean(1);    // [B=1, 1, S_kv]
  attn = attn.squeeze();  // [S_kv]

  // Extract attention from current token to visual tokens
  // [S_img]
  auto text_2_visual = attn[{{img_token_start, img_token_end}}].contiguous();

  // Calculate the topk tokens number
  MLLM_RT_ASSERT_EQ(text_2_visual.rank(), 1);
  auto S_img_tokens = text_2_visual.shape()[0];
  int32_t left_tokens = (int32_t)std::ceil(S_img_tokens * (1 - pruning_ratio));

  auto [_, indices] = mllm::nn::functional::topk(text_2_visual, left_tokens, /*dim*/ -1, /*largest*/ true, /*sorted*/ false);

  return indices.toVector<int32_t>();
}
