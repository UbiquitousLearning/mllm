// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "BenchmarkTemplate.hpp"

class Qwen3_W4A32_KAI_Benchmark final : public BenchmarkTemplate {
 public:
  void init(const std::string& cfg_path, const std::string& model_path, int32_t cache_length) override {}

  void printModelInfo() override {}

  void warmup() override {}

  void clear() override {}

  BenchmarkTemplateResult run(int32_t pp, int32_t tg) override {
    // TODO

    return {};
  }

 private:
};
