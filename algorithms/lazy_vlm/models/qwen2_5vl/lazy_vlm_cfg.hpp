// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <mllm/mllm.hpp>

struct LazyVLMConfig {
  bool decode_callback = true;
  // clang-format off
  std::unordered_map<int32_t, float> pruning_settings = {
    {3, 0.15},
    {6, 0.15},
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
};
