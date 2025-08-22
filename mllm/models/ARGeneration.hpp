// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "mllm/core/Tensor.hpp"
#include "mllm/utils/AnyValue.hpp"

namespace mllm::models {

using ARGenerationOutputPast = std::unordered_map<std::string, Tensor>;
using ARGenerationArgs = std::unordered_map<std::string, AnyValue>;

class ARGeneration {
 public:
  virtual ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) = 0;

  virtual ARGenerationOutputPast generate(const ARGenerationOutputPast& input, const ARGenerationArgs& args);

  virtual void streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                              const std::function<void(int64_t)>& callback);

  int64_t sampleGreedy(Tensor& logits);

  int64_t sampleTemperature(Tensor& logits, float temperature);

  int64_t sampleTopK(Tensor& logits, int k, float temperature);

  int64_t sampleTopP(Tensor& logits, float p, float temperature);

  int64_t categoricalSample(const Tensor& probs);

  Tensor getLastLogits(Tensor& logits);

  int sampleFromDistribution(const std::vector<float>& probs);

 protected:
  bool do_sample_ = false;
  int eos_token_id_;
  int max_length_ = 1024;
};

}  // namespace mllm::models
