// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <chrono>
#include <string>

#include "BenchmarkTemplate.hpp"

#include <mllm/mllm.hpp>
#include <mllm/models/llama/modeling_llama.hpp>
#include <mllm/models/llama/configuration_llama.hpp>

class Llama_Benchmark final : public BenchmarkTemplate {
 public:
  std::optional<KVCacheEstimateInfo> kvEstimateInfo() const override {
    if (!cfg_) return std::nullopt;
    KVCacheEstimateInfo info;
    info.num_layers = cfg_->num_hidden_layers;
    info.num_kv_heads = cfg_->num_key_value_heads;
    info.head_dim = cfg_->hidden_size / cfg_->num_attention_heads;
    return info;
  }

  void init(const std::string& cfg_path, const std::string& model_path, int32_t cache_length) override {
    cfg_ = std::make_unique<mllm::models::llama::LLaMAConfig>(cfg_path);

    // LLaMA config uses max_position_embeddings as KV-cache upper bound
    if (cache_length > 0) { cfg_->max_position_embeddings = cache_length; }

    model_ = std::make_unique<mllm::models::llama::LlamaForCausalLM>("", *cfg_);

    // V1 param file only
    auto param = mllm::load(model_path, mllm::ModelFileVersion::kV1);
    model_->load(param);

    mllm::print("Model initialized successfully");
  }

  void printModelInfo() override {
    if (!cfg_) return;
    mllm::print("========== Model Information ==========");
    mllm::print("Model Type         : LLaMA / TinyLlama");
    mllm::print("Hidden Size        :", cfg_->hidden_size);
    mllm::print("Num Layers         :", cfg_->num_hidden_layers);
    mllm::print("Num Heads          :", cfg_->num_attention_heads);
    mllm::print("Num KV Heads       :", cfg_->num_key_value_heads);
    int32_t head_dim = (cfg_->num_attention_heads > 0) ? (cfg_->hidden_size / cfg_->num_attention_heads) : 0;
    mllm::print("Head Dim           :", head_dim);
    mllm::print("Intermediate Size  :", cfg_->intermediate_size);
    mllm::print("Vocab Size         :", cfg_->vocab_size);
    mllm::print("Max Pos Embeddings :", cfg_->max_position_embeddings);
    mllm::print("=======================================");
  }

  void warmup() override {
    if (!model_) return;

    const int32_t warmup_length = 8;
    const int32_t warmup_gen = 4;

    auto input_ids = mllm::Tensor::empty({1, warmup_length}, mllm::kInt64, mllm::kCPU).setMemType(mllm::kNormal).alloc();
    auto ptr = input_ids.ptr<mllm::mllm_int64_t>();
    for (int i = 0; i < warmup_length; ++i) ptr[i] = 1;

    mllm::models::ARGenerationOutputPast inputs;
    inputs["sequence"] = input_ids;

    mllm::models::ARGenerationArgs args;
    args["max_length"] = mllm::AnyValue((int)warmup_gen);
    args["do_sample"] = mllm::AnyValue(false);

    model_->generate(inputs, args);
    mllm::print("Warmup completed");
  }

  void clear() override {
    if (!model_) {
      return;
    }
    model_->kvCache().setCurrentSeqCnt(0);
  }

  BenchmarkTemplateResult run(int32_t pp, int32_t tg) override {
    if (pp <= 0 || tg < 0) {
      mllm::print("[ERROR] invalid pp/tg:", pp, tg);
      return {0.f, 0.f, 0.f};
    }
    if (!model_) return {0.f, 0.f, 0.f};

    auto input_ids = mllm::Tensor::empty({1, pp}, mllm::kInt64, mllm::kCPU).setMemType(mllm::kNormal).alloc();
    auto ptr = input_ids.ptr<mllm::mllm_int64_t>();
    for (int i = 0; i < pp; ++i) ptr[i] = 1 + (i % 100);

    mllm::models::ARGenerationOutputPast inputs;
    inputs["sequence"] = input_ids;

    mllm::models::ARGenerationArgs args;
    args["max_length"] = mllm::AnyValue((int)tg);
    args["do_sample"] = mllm::AnyValue(false);

    auto prefill_start = std::chrono::high_resolution_clock::now();
    auto decode_start = prefill_start;
    auto decode_end = prefill_start;

    bool first_token = true;
    int token_count = 0;

    model_->streamGenerate(inputs, args, [&](int64_t /*token_id*/) {
      if (first_token) {
        decode_start = std::chrono::high_resolution_clock::now();
        first_token = false;
      }
      token_count++;
      decode_end = std::chrono::high_resolution_clock::now();
    });

    auto prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_start - prefill_start).count();
    auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();

    BenchmarkTemplateResult r;
    r.ttft = prefill_us / 1000.0f;
    r.prefill_speed = (prefill_us > 0) ? (static_cast<float>(pp) / prefill_us) * 1e6f : 0.f;
    // exclude first token from decode throughput
    int decode_tokens = (token_count > 0) ? (token_count - 1) : 0;
    r.decode_speed = (decode_us > 0 && decode_tokens > 0) ? (static_cast<float>(decode_tokens) / decode_us) * 1e6f : 0.f;
    return r;
  }

 private:
  std::unique_ptr<mllm::models::llama::LLaMAConfig> cfg_;
  std::unique_ptr<mllm::models::llama::LlamaForCausalLM> model_;
};
