// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>

struct BenchmarkTemplateResult {
  float ttft;
  float prefill_speed;
  float decode_speed;
};

class BenchmarkTemplate {
 public:
  virtual void init(const std::string& cfg_path, const std::string& model_path, int32_t cache_length) = 0;

  virtual void printModelInfo() = 0;

  virtual void warmup() = 0;

  virtual void clear() = 0;

  virtual BenchmarkTemplateResult run(int32_t pp, int32_t tg) = 0;
};
