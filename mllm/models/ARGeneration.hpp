// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iterator>

#include "mllm/core/Tensor.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::models {

using ARGenerationOutputPast = std::unordered_map<std::string, Tensor>;
using ARGenerationArgs = std::unordered_map<std::string, AnyValue>;
using IROutput = std::unordered_map<std::string, ir::IRContext::ptr_t>;

struct ARGenerationStep {
  int64_t current_step = -1;
  int64_t cur_token_id = 0;
};

class ARGeneration;
struct ARGenerationChatContext;

class ARGenerationChatIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = ARGenerationStep;
  using difference_type = std::ptrdiff_t;
  using pointer = const ARGenerationStep*;
  using reference = const ARGenerationStep&;

  ARGenerationChatIterator(ARGeneration& gen, const ARGenerationOutputPast& initial_input, const ARGenerationArgs& args);

  ARGenerationChatIterator();

  reference operator*() const;

  pointer operator->() const;

  ARGenerationChatIterator& operator++();

  bool operator==(const ARGenerationChatIterator& other) const;

  bool operator!=(const ARGenerationChatIterator& other) const;

 private:
  void step();

  ARGeneration* gen_ = nullptr;
  ARGenerationOutputPast current_input_;
  ARGenerationArgs args_;
  ARGenerationStep current_step_;
  bool finished_ = true;
  int64_t step_count_ = 0;

  float temperature_;
  int top_k_;
  float top_p_;
  int max_length_;
  int eos_token_id_;
  bool do_sample_;
};

struct ARGenerationChatContext {
  ARGenerationChatContext(ARGeneration& gen, const ARGenerationOutputPast& input, const ARGenerationArgs& args);

  ARGenerationChatIterator begin();

  ARGenerationChatIterator end();

 private:
  ARGeneration& gen_;
  ARGenerationOutputPast input_;
  ARGenerationArgs args_;
};

class ARGeneration {
 public:
  friend struct ARGenerationChatIterator;
  friend struct ARGenerationChatContext;

  virtual ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) = 0;

  virtual ARGenerationOutputPast generate(const ARGenerationOutputPast& input, const ARGenerationArgs& args);

  virtual void streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                              const std::function<void(int64_t)>& callback);

  virtual IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args);

  int64_t sampleGreedy(Tensor& logits);

  int64_t sampleTemperature(Tensor& logits, float temperature);

  int64_t sampleTopK(Tensor& logits, int k, float temperature);

  int64_t sampleTopP(Tensor& logits, float p, float temperature);

  ARGenerationChatContext chat(const ARGenerationOutputPast& input, const ARGenerationArgs& args = {});

  int64_t categoricalSample(const Tensor& probs);

  Tensor getLastLogits(Tensor& logits);

  int sampleFromDistribution(const std::vector<float>& probs);

 protected:
  bool do_sample_ = false;
  int eos_token_id_;
  int max_length_ = 1024;
};

}  // namespace mllm::models
