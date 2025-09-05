// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <ranges>
#include <algorithm>
#include <unordered_map>
#include <mllm/mllm.hpp>
#include <mllm/nn/Functional.hpp>

struct LazyVLMConfig {
  bool decode_callback = true;
  // clang-format off
  std::unordered_map<int32_t, float> pruning_settings = {
    {3, 0.2},
    {6, 0.2},
    {9, 0.2},
    {12, 0.2},
    {15, 0.2},
    {18, 0.2},
  };
  // clang-format on
};

struct LazyVLMState {
  // AR Generation info
  int32_t cur_step = -1;

  // You can get visual tokens by doing [self.first_img_token_pos:self.last_img_token_pos]
  int32_t first_img_token_pos = 0;
  int32_t last_img_token_pos = 0;

  mllm::Tensor llm_prefill_sin = mllm::Tensor::nil();
  mllm::Tensor llm_prefill_cos = mllm::Tensor::nil();

  mllm::Tensor llm_current_sin = mllm::Tensor::nil();
  mllm::Tensor llm_current_cos = mllm::Tensor::nil();

  // Record the chosen position in each layer in each step.
  std::vector<std::vector<int>> chosen_pos_in_each;
  std::vector<std::vector<int>> chosen_pos_to_delay_compute;

  inline std::tuple<mllm::Tensor, int, int> attention_score_analyze_prefill(mllm::Tensor attn, int32_t layer_idx) {
    // Prefill attention inputs is [B=1, H, S_all, S_all]
    attn = attn.mean(1);     // [B=1, S_all, S_all]
    attn = attn.squeeze(0);  // [S_all, S_all]

    auto& cur_chosen_tokens = chosen_pos_in_each[layer_idx];
    auto visual_start_in_selected = std::ranges::find(cur_chosen_tokens, first_img_token_pos - 1);
    auto visual_end_in_selected = std::ranges::find(cur_chosen_tokens, last_img_token_pos);
    assert(visual_start_in_selected != cur_chosen_tokens.end() && visual_end_in_selected != cur_chosen_tokens.end());
    int visual_start = std::distance(cur_chosen_tokens.begin(), visual_start_in_selected);
    int visual_end = std::distance(cur_chosen_tokens.begin(), visual_end_in_selected);

    // [S_text, S_img]
    auto text_2_visual = attn[{{visual_end, mllm::kAll}, {visual_start + 1, visual_end}}].contiguous();
    text_2_visual = text_2_visual.sum(0);  // [S_img]

    return {text_2_visual, visual_start, visual_end};
  }

  inline std::tuple<mllm::Tensor, int, int> attention_score_analyze_decode(mllm::Tensor attn, int layer_idx) {
    // Decode attention inputs is [B=1, H, suppose 1, S_kv]
    // For decode phase, we have attention from current token to all previous tokens including visual tokens
    attn = attn.mean(1);     // [B=1, suppose 1, S_kv]
    attn = attn.squeeze(0);  // [suppose 1, S_kv]

    if (attn.shape()[1] != 1) {
      // Attn's length is not 1 means there has delay compute tokens! When calculate attn score in decoding stage, we just
      // consider the generated tokens' contribution to visual tokens. So we need to slice it
      auto d = attn.shape()[1];
      attn = attn[{-1, mllm::kAll}].contiguous().view({1, d});
    }

    auto& cur_chosen_tokens = chosen_pos_in_each[layer_idx];
    auto visual_start_in_selected = std::ranges::find(cur_chosen_tokens, first_img_token_pos - 1);
    auto visual_end_in_selected = std::ranges::find(cur_chosen_tokens, last_img_token_pos);
    assert(visual_start_in_selected != cur_chosen_tokens.end() && visual_end_in_selected != cur_chosen_tokens.end());
    int visual_start = std::distance(cur_chosen_tokens.begin(), visual_start_in_selected);
    int visual_end = std::distance(cur_chosen_tokens.begin(), visual_end_in_selected);

    attn = attn[{mllm::kAll, {visual_start + 1, visual_end}}].squeeze(0);
    return {attn, visual_start, visual_end};
  }

  inline std::vector<int> select_high_score_visual_token_prefill(const mllm::Tensor& attn, int layer_idx, float pruning_rate) {
    // Get current chosen pos
    auto& cur_chosen_pos = chosen_pos_in_each[layer_idx];

    // Get attention score. visual_start, visual_end.
    auto [attn_score, v_s, v_e] = attention_score_analyze_prefill(attn, layer_idx);

    auto cur_visual_token_length = attn_score.shape()[0];
    assert((v_e - v_s - 1) == cur_visual_token_length);

    // Make cur_keep_visual_tokens dividable by chunk size
    auto keep_ratio = 1 - pruning_rate;
    int k_initial = std::ceil(cur_visual_token_length * keep_ratio);
    auto k_final = std::min(k_initial, cur_visual_token_length);
    assert((0 <= k_final) && (k_final <= cur_visual_token_length));

    // Use TopK to get the position of the tokens we need
    auto [_, topk_indices_tensor] = mllm::nn::functional::topk(attn_score, k_final);
    auto topk_indices = topk_indices_tensor.toVector<int>();

    // Select the visual tokens we need.
    std::vector<int> selected_visual_tokens_pos;
    std::ranges::transform(topk_indices, std::back_inserter(selected_visual_tokens_pos),
                           [&](int item) { return cur_chosen_pos[v_s + 1 + item]; });
    std::vector<int> final_token_chosen;
    auto part1 = cur_chosen_pos | std::views::take(v_s + 1);
    auto part3 = cur_chosen_pos | std::views::drop(v_e);

    // Final tokens chosen
    final_token_chosen.reserve(part1.size() + selected_visual_tokens_pos.size() + part3.size());
    std::ranges::copy(part1, std::back_inserter(final_token_chosen));
    std::ranges::copy(selected_visual_tokens_pos, std::back_inserter(final_token_chosen));
    std::ranges::copy(part3, std::back_inserter(final_token_chosen));
    std::ranges::sort(final_token_chosen);

    // NOTE: The token position here is still the index number in the original sequence.
    assert(cur_chosen_pos.size() - (cur_visual_token_length - k_final) == final_token_chosen.size());
    return final_token_chosen;
  }

  inline std::vector<int> select_high_score_visual_token_decode(const mllm::Tensor& attn, int layer_idx, float pruning_rate) {
    // Get current chosen pos
    auto& cur_chosen_pos = chosen_pos_in_each[layer_idx];

    // Get attention score. visual_start, visual_end.
    auto [attn_score, v_s, v_e] = attention_score_analyze_decode(attn, layer_idx);

    auto cur_visual_token_length = attn_score.shape()[0];
    assert((v_e - v_s - 1) == cur_visual_token_length);

    // Make cur_keep_visual_tokens dividable by chunk size
    auto keep_ratio = 1 - pruning_rate;
    int k_initial = std::ceil(cur_visual_token_length * keep_ratio);
    auto k_final = std::min(k_initial, cur_visual_token_length);
    assert((0 <= k_final) && (k_final <= cur_visual_token_length));

    // Use TopK to get the position of the tokens we need
    auto [_, topk_indices_tensor] = mllm::nn::functional::topk(attn_score, k_final);
    auto topk_indices = topk_indices_tensor.toVector<int>();

    // Select the visual tokens we need.
    std::vector<int> selected_visual_tokens_pos;
    std::ranges::transform(topk_indices, std::back_inserter(selected_visual_tokens_pos),
                           [&](int item) { return cur_chosen_pos[v_s + 1 + item]; });
    std::vector<int> final_token_chosen;
    auto part1 = cur_chosen_pos | std::views::take(v_s + 1);
    auto part3 = cur_chosen_pos | std::views::drop(v_e);

    // Final tokens chosen
    final_token_chosen.reserve(part1.size() + selected_visual_tokens_pos.size() + part3.size());
    std::ranges::copy(part1, std::back_inserter(final_token_chosen));
    std::ranges::copy(selected_visual_tokens_pos, std::back_inserter(final_token_chosen));
    std::ranges::copy(part3, std::back_inserter(final_token_chosen));
    std::ranges::sort(final_token_chosen);

    // NOTE: The token position here is still the index number in the original sequence.
    assert(cur_chosen_pos.size() - (cur_visual_token_length - k_final) == final_token_chosen.size());
    return final_token_chosen;
  }
};
