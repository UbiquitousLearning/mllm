// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <random>
#include <algorithm>

#include "mllm/nn/Functional.hpp"
#include "mllm/utils/UnsafeMacros.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models {

ARGenerationOutputPast ARGeneration::generate(const ARGenerationOutputPast& input, const ARGenerationArgs& args) {
  ARGenerationOutputPast past = input;
  std::vector<int64_t> generated_tokens;

  // Init values
  float temperature = args.count("temperature") ? args.at("temperature").get<float>() : 1.0f;
  int top_k = args.count("top_k") ? args.at("top_k").get<int>() : 0;
  float top_p = args.count("top_p") ? args.at("top_p").get<float>() : 0.0f;
  int max_length = args.count("max_length") ? args.at("max_length").get<int>() : max_length_;
  int eos_token_id = args.count("eos_token_id") ? args.at("eos_token_id").get<int>() : eos_token_id_;
  bool do_sample = args.count("do_sample") ? args.at("do_sample").get<bool>() : do_sample_;

  bool use_sampling = do_sample || (temperature != 1.0f) || (top_k > 0) || (top_p > 0.0f);

  for (int i = 0; i < max_length; ++i) {
    ARGenerationOutputPast output = forward(past, args);
    Tensor logits = output["logits"];

    int64_t next_token_id;
    if (use_sampling) {
      if (top_k > 0) {
        next_token_id = sampleTopK(logits, top_k, temperature);
      } else if (top_p > 0.0f) {
        next_token_id = sampleTopP(logits, top_p, temperature);
      } else {
        next_token_id = sampleTemperature(logits, temperature);
      }
    } else {
      next_token_id = sampleGreedy(logits);
    }

    generated_tokens.push_back(next_token_id);

    if (next_token_id == eos_token_id) { break; }

    // Clear things in output.
    output["logits"] = Tensor::nil();

    // [B, S]
    past = output;
    past["input_ids"] = Tensor::empty({1, 1}, kInt64, logits.device()).alloc();
    past["input_ids"].at<mllm_int64_t>({0}) = next_token_id;
  }

  // From blob
  Tensor generated_tensor = Tensor::empty({(int32_t)generated_tokens.size()}, kInt64, kCPU).alloc();
  __MLLM_UNSAFE_OPT_BEGIN_O3
  std::copy(generated_tokens.begin(), generated_tokens.end(), generated_tensor.ptr<mllm_int64_t>());
  __MLLM_UNSAFE_OPT_END

  // Clear things in output.
  past["logits"] = Tensor::nil();
  past["input_ids"] = Tensor::nil();
  past["generated_sequence"] = generated_tensor;
  return past;
}

void ARGeneration::streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                                  const std::function<void(int64_t)>& callback) {
  ARGenerationOutputPast past = input;

  // Init values
  float temperature = args.count("temperature") ? args.at("temperature").get<float>() : 1.0f;
  int top_k = args.count("top_k") ? args.at("top_k").get<int>() : 0;
  float top_p = args.count("top_p") ? args.at("top_p").get<float>() : 0.0f;
  int max_length = args.count("max_length") ? args.at("max_length").get<int>() : max_length_;
  int eos_token_id = args.count("eos_token_id") ? args.at("eos_token_id").get<int>() : eos_token_id_;
  bool do_sample = args.count("do_sample") ? args.at("do_sample").get<bool>() : do_sample_;

  bool use_sampling = do_sample || (temperature != 1.0f) || (top_k > 0) || (top_p > 0.0f);

  for (int i = 0; i < max_length; ++i) {
    ARGenerationOutputPast output = forward(past, args);
    Tensor logits = output["logits"];

    int64_t next_token_id;
    if (use_sampling) {
      if (top_k > 0) {
        next_token_id = sampleTopK(logits, top_k, temperature);
      } else if (top_p > 0.0f) {
        next_token_id = sampleTopP(logits, top_p, temperature);
      } else {
        next_token_id = sampleTemperature(logits, temperature);
      }
    } else {
      next_token_id = sampleGreedy(logits);
    }

    callback(next_token_id);

    if (next_token_id == eos_token_id) { break; }

    // Clear things in output.
    output["logits"] = Tensor::nil();

    // [B, S]
    past = output;
    past["input_ids"] = Tensor::empty({1, 1}, kInt64, logits.device()).alloc();
    past["input_ids"].at<mllm_int64_t>({0}) = next_token_id;
  }
}

int64_t ARGeneration::sampleGreedy(Tensor& logits) {
  Tensor last_logits = getLastLogits(logits);

  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(last_logits.dtype(), kFloat32);

  auto logits_data = last_logits.ptr<float>();
  int vocab_size = last_logits.shape().back();
  __MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
  auto max_it = std::max_element(logits_data, logits_data + vocab_size);
  __MLLM_UNSAFE_OPT_END
  return std::distance(logits_data, max_it);
}

int64_t ARGeneration::sampleTemperature(Tensor& logits, float temperature) {
  Tensor last_logits = getLastLogits(logits);

  if (temperature != 1.0f) { last_logits = last_logits * (1.f / temperature); }

  Tensor probs = nn::functional::softmax(last_logits, -1);

  // I NEED MORE MEMORY!
  delete &last_logits;
  delete &logits;

  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);

  return categoricalSample(probs);
}

int64_t ARGeneration::sampleTopK(Tensor& logits, int k, float temperature) {
  // We need the last logits
  Tensor last_logits = getLastLogits(logits);

  // Apply temperature.
  if (temperature != 1.0f) { last_logits = last_logits * (1.f / temperature); }

  // Get Softmax probabilities
  Tensor probs = nn::functional::softmax(last_logits, -1);

  // I NEED MORE MEMORY!
  delete &last_logits;
  delete &logits;

  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);
  auto prob_data = probs.ptr<float>();
  int vocab_size = probs.shape().back();

  __MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
  // Indexing
  std::vector<int> indices(vocab_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&prob_data](int i1, int i2) { return prob_data[i1] > prob_data[i2]; });
  // Get top k
  std::vector<float> top_k_probs(k);
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    top_k_probs[i] = prob_data[indices[i]];
    sum += top_k_probs[i];
  }
  // Norm probs
  for (int i = 0; i < k; ++i) { top_k_probs[i] *= (1.f / sum); }
  __MLLM_UNSAFE_OPT_END

  return indices[sampleFromDistribution(top_k_probs)];
}

int64_t ARGeneration::sampleTopP(Tensor& logits, float p, float temperature) {
  Tensor last_logits = getLastLogits(logits);

  if (temperature != 1.0f) { last_logits = last_logits * (1.f / temperature); }

  Tensor probs = nn::functional::softmax(last_logits, -1);

  // I NEED MORE MEMORY!
  delete &last_logits;
  delete &logits;

  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);
  auto prob_data = probs.ptr<float>();
  int vocab_size = probs.shape().back();

  __MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
  std::vector<int> indices(vocab_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&prob_data](int i1, int i2) { return prob_data[i1] > prob_data[i2]; });
  std::vector<float> top_probs;
  float cumulative_prob = 0.0f;
  int i = 0;
  for (; i < vocab_size && cumulative_prob < p; ++i) {
    top_probs.push_back(prob_data[indices[i]]);
    cumulative_prob += prob_data[indices[i]];
  }
  float sum = std::accumulate(top_probs.begin(), top_probs.end(), 0.0f);
  for (float& prob : top_probs) { prob *= (1.f / sum); }
  __MLLM_UNSAFE_OPT_END

  return indices[sampleFromDistribution(top_probs)];
}

int64_t ARGeneration::categoricalSample(const Tensor& probs) {
  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);
  auto prob_data = probs.ptr<float>();
  int vocab_size = probs.shape().back();

  __MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
  std::vector<float> cumulative_probs(vocab_size);
  std::partial_sum(prob_data, prob_data + vocab_size, cumulative_probs.begin());
  __MLLM_UNSAFE_OPT_END

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  float r = dis(gen);

  auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), r);
  return std::distance(cumulative_probs.begin(), it);
}

Tensor ARGeneration::getLastLogits(Tensor& logits) {
  // [B, S, D] for almost all TextLLM, VLM.
  if (logits.shape().size() == 3) {
    if (logits.shape()[1] == 1) {
      return logits;
    } else {
      // Get the last logits
      return logits[{
          kAll,
          logits.shape()[1] - 1,
          kAll,
      }];
    }
  } else {
    throw std::runtime_error("getLastLogits suppose inputs logits is [B, S, D] layout.");
  }
}

int ARGeneration::sampleFromDistribution(const std::vector<float>& probs) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return dist(gen);
}

}  // namespace mllm::models
