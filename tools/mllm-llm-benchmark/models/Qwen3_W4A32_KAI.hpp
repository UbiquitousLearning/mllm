// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <chrono>
#include <thread>

#include "BenchmarkTemplate.hpp"
#include <mllm/mllm.hpp>
#include <mllm/models/qwen3/modeling_qwen3.hpp>
#include <mllm/models/qwen3/configuration_qwen3.hpp>

class Qwen3_W4A32_KAI_Benchmark final : public BenchmarkTemplate {
 public:
  void init(const std::string& cfg_path, const std::string& model_path, int32_t cache_length) override {
    // Load config
    config_ = std::make_unique<mllm::models::qwen3::Qwen3Config>(cfg_path);
    
    // Override cache length if specified
    if (cache_length > 0) {
      config_->max_cache_length = cache_length;
    }
    
    // Create model
    model_ = std::make_unique<mllm::models::qwen3::Qwen3ForCausalLM>(*config_);
    
    // Load weights
    auto param = mllm::load(model_path, mllm::ModelFileVersion::kV2);
    model_->load(param);
    
    mllm::print("Model initialized successfully");
  }

  void printModelInfo() override {
    if (!config_) {
      mllm::print("Config not loaded");
      return;
    }
    
    mllm::print("========== Model Information ==========");
    mllm::print("Model Type         : Qwen3 W4A32 KAI");
    mllm::print("Hidden Size        :", config_->hidden_size);
    mllm::print("Num Layers         :", config_->num_hidden_layers);
    mllm::print("Num Heads          :", config_->num_attention_heads);
    mllm::print("Num KV Heads       :", config_->num_key_value_heads);
    mllm::print("Head Dim           :", config_->head_dim);
    mllm::print("Intermediate Size  :", config_->intermediate_size);
    mllm::print("Vocab Size         :", config_->vocab_size);
    mllm::print("Max Cache Length   :", config_->max_cache_length);
    mllm::print("Linear Impl        :", static_cast<int>(config_->linear_impl_type));
    mllm::print("=======================================");
  }

  void warmup() override {
    if (!model_) {
      mllm::print("Model not initialized");
      return;
    }
    
    // Create a small dummy input for warmup (e.g., 8 tokens)
    const int32_t warmup_length = 8;
    const int32_t warmup_gen = 4;
    
    mllm::print("Warming up with", warmup_length, "tokens prefill and", warmup_gen, "tokens generation...");
    
    // Create input with proper memory type
    auto input_ids = mllm::Tensor::empty({1, warmup_length}, mllm::kInt64, mllm::kCPU)
                         .setMemType(mllm::kNormal)
                         .alloc();
    auto ptr = input_ids.ptr<mllm::mllm_int64_t>();
    for (int i = 0; i < warmup_length; ++i) {
      ptr[i] = 1;  // Use token id 1
    }
    
    mllm::models::ARGenerationOutputPast inputs;
    inputs["sequence"] = input_ids;
    
    mllm::models::ARGenerationArgs args;
    int max_len = warmup_gen;
    bool do_sample = false;
    args["max_length"] = mllm::AnyValue(max_len);
    args["do_sample"] = mllm::AnyValue(do_sample);
    
    // Run warmup
    model_->generate(inputs, args);
    
    mllm::print("Warmup completed");
  }

  void clear() override {
    if (!model_) {
      return;
    }
    
    // Clear KV cache by resetting sequence count
    model_->kvCache().setCurrentSeqCnt(0);
  }

  BenchmarkTemplateResult run(int32_t pp, int32_t tg) override {
    if (!model_) {
      mllm::print("Model not initialized");
      return {0.0f, 0.0f, 0.0f};
    }
    
    // Create input with specified prompt length
    auto input_ids = mllm::Tensor::empty({1, pp}, mllm::kInt64, mllm::kCPU)
                         .setMemType(mllm::kNormal)
                         .alloc();
    auto ptr = input_ids.ptr<mllm::mllm_int64_t>();
    for (int i = 0; i < pp; ++i) {
      ptr[i] = 1 + (i % 100);  // Varied token ids
    }
    
    mllm::models::ARGenerationOutputPast inputs;
    inputs["sequence"] = input_ids;
    
    mllm::models::ARGenerationArgs args;
    int max_len = tg;
    bool do_sample = false;
    args["max_length"] = mllm::AnyValue(max_len);
    args["do_sample"] = mllm::AnyValue(do_sample);
    
    // Use streamGenerate with manual timing
    auto prefill_start = std::chrono::high_resolution_clock::now();
    auto decode_start = std::chrono::high_resolution_clock::now();
    auto decode_end = std::chrono::high_resolution_clock::now();
    
    bool first_token = true;
    int token_count = 0;
    
    model_->streamGenerate(inputs, args, [&](int64_t token_id) {
      if (first_token) {
        decode_start = std::chrono::high_resolution_clock::now();
        first_token = false;
      }
      token_count++;
      decode_end = std::chrono::high_resolution_clock::now();
    });
    
    // Calculate durations
    auto prefill_duration = std::chrono::duration_cast<std::chrono::microseconds>(decode_start - prefill_start).count();
    auto decode_duration = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();
    auto ttft_duration = prefill_duration;
    
    BenchmarkTemplateResult result;
    
    // TTFT in milliseconds
    result.ttft = ttft_duration / 1000.0f;
    
    // Prefill speed in tokens/s
    if (prefill_duration > 0) {
      result.prefill_speed = (static_cast<float>(pp) / prefill_duration) * 1000000.0f;
    } else {
      result.prefill_speed = 0.0f;
    }
    
    // Decode speed in tokens/s
    if (decode_duration > 0 && token_count > 0) {
      result.decode_speed = (static_cast<float>(token_count) / decode_duration) * 1000000.0f;
    } else {
      result.decode_speed = 0.0f;
    }
    
    return result;
  }

 private:
  std::unique_ptr<mllm::models::qwen3::Qwen3Config> config_;
  std::unique_ptr<mllm::models::qwen3::Qwen3ForCausalLM> model_;
};
