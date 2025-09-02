// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

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

struct LazyVLMSate {};
