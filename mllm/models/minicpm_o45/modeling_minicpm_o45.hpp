// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/llama/configuration_llama.hpp"
#include "mllm/models/llama/modeling_llama.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/modeling_resampler.hpp"
#include "mllm/models/minicpm_o2_6/modeling_siglip.hpp"
#include "mllm/models/minicpm_o2_6/modeling_whisper_encoder.hpp"
#include "mllm/models/minicpm_o45/configuration_minicpm_o45.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/modeling_qwen3.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::minicpm_o45 {

class AudioProjectionLayer final : public nn::Module {
 public:
  AudioProjectionLayer() = default;

  AudioProjectionLayer(const std::string& name, int32_t input_dim, int32_t hidden_dim, int32_t output_dim) : Module(name) {
    linear1_ = reg<nn::Linear>("linear1", input_dim, hidden_dim, true);
    relu_ = reg<nn::ReLU>("relu");
    linear2_ = reg<nn::Linear>("linear2", hidden_dim, output_dim, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    x = linear1_(x);
    x = relu_(x);
    x = linear2_(x);
    return {x};
  }

 private:
  nn::Linear linear1_;
  nn::ReLU relu_;
  nn::Linear linear2_;
};

class AudioAvgPooler final : public nn::Module {
 public:
  AudioAvgPooler() = default;

  AudioAvgPooler(const std::string& name, int32_t kernel_size, int32_t stride) : Module(name) {
    avg_pool_ = reg<nn::AvgPool1d>("pool", kernel_size, stride);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {avg_pool_(inputs[0])};
  }

 private:
  nn::AvgPool1d avg_pool_;
};

class TTSProjector final : public nn::Module {
 public:
  TTSProjector() = default;

  TTSProjector(const std::string& name, int32_t input_dim, int32_t output_dim) : nn::Module(name) {
    linear1_ = reg<nn::Linear>("linear1", input_dim, output_dim, true);
    relu_ = reg<nn::ReLU>("relu");
    linear2_ = reg<nn::Linear>("linear2", output_dim, output_dim, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = linear1_(inputs[0]);
    x = relu_(x);
    x = linear2_(x);
    return {x};
  }

 private:
  nn::Linear linear1_;
  nn::ReLU relu_;
  nn::Linear linear2_;
};

struct MiniCPMO45TTSGenerationConfig {
  int32_t max_new_tokens = 1024;
  int32_t min_new_tokens = 50;
  bool force_no_stop = false;
  bool do_sample = true;
  int32_t top_k = 25;
  float top_p = 0.85f;
  float repetition_penalty = 1.05f;
  int32_t repetition_penalty_window = 16;
  std::vector<float> temperature = {0.8f};
  int32_t debug_interval = 16;
  std::function<void(int32_t step, const std::vector<int64_t>& tokens, bool has_eos)> step_callback = nullptr;
};

struct MiniCPMO45TTSGenerationOutput {
  Tensor new_ids = Tensor::nil();
  bool finished = false;
};

class MiniCPMO45TTS final : public nn::Module {
 public:
  MiniCPMO45TTS() = default;

  MiniCPMO45TTS(const std::string& name, const MiniCPMO45Config& cfg) : nn::Module(name), cfg_(cfg) {
    projector_spk_ = reg<TTSProjector>("projector_spk", cfg.tts_llm_dim, cfg.tts_hidden_size);
    projector_semantic_ = reg<TTSProjector>("projector_semantic", cfg.tts_llm_dim, cfg.tts_hidden_size);

    emb_text_ = reg<nn::Embedding>("emb_text", cfg.tts_num_text_tokens, cfg.tts_hidden_size);

    emb_code_.reserve(cfg.tts_num_vq);
    for (int32_t i = 0; i < cfg.tts_num_vq; ++i) {
      emb_code_.emplace_back(reg<nn::Embedding>("emb_code." + std::to_string(i), cfg.tts_num_audio_tokens, cfg.tts_hidden_size));
    }

    auto llama_cfg = llama::LLaMAConfig();
    llama_cfg.vocab_size = cfg.tts_backbone_vocab_size;
    llama_cfg.hidden_size = cfg.tts_hidden_size;
    llama_cfg.intermediate_size = cfg.tts_intermediate_size;
    llama_cfg.num_attention_heads = cfg.tts_num_attention_heads;
    llama_cfg.num_key_value_heads = cfg.tts_num_key_value_heads;
    llama_cfg.num_hidden_layers = cfg.tts_num_hidden_layers;
    llama_cfg.max_position_embeddings = cfg.tts_max_position_embeddings;
    llama_cfg.rms_norm_eps = cfg.tts_rms_norm_eps;
    llama_cfg.rope_theta = cfg.tts_rope_theta;
    llama_cfg.hidden_act = cfg.tts_hidden_act;
    llama_cfg.tie_word_embeddings = false;
    llama_cfg.attention_bias = false;
    llama_cfg.linear_impl_type = cfg.linear_impl_type;
    model_ = reg<llama::LlamaText>("model", llama_cfg);
  }

  void loadFromParameter(const ParameterFile::ptr_t& param_file) {
    nn::Module::load(param_file);

    head_code_weight_.clear();
    head_code_weight_.reserve(cfg_.tts_num_vq);

    auto prefix = getModuleName() + ".head_code.";
    for (int32_t i = 0; i < cfg_.tts_num_vq; ++i) {
      auto g = param_file->pull(prefix + std::to_string(i) + ".parametrizations.weight.original0");
      auto v = param_file->pull(prefix + std::to_string(i) + ".parametrizations.weight.original1");
      if (g.dtype() != kFloat32) { g = g.to(kFloat32); }
      if (v.dtype() != kFloat32) { v = v.to(kFloat32); }
      g = g.contiguous();
      v = v.contiguous().view({cfg_.tts_num_audio_tokens, cfg_.tts_hidden_size});

      auto weight = Tensor::empty({cfg_.tts_num_audio_tokens, cfg_.tts_hidden_size}, kFloat32, kCPU).alloc();

      auto* g_ptr = g.ptr<float>();
      auto* v_ptr = v.ptr<float>();
      auto* w_ptr = weight.ptr<float>();

      constexpr float kEps = 1e-12f;
      for (int32_t out_idx = 0; out_idx < cfg_.tts_num_audio_tokens; ++out_idx) {
        float norm = 0.0f;
        auto row_offset = out_idx * cfg_.tts_hidden_size;
        for (int32_t d = 0; d < cfg_.tts_hidden_size; ++d) {
          auto val = v_ptr[row_offset + d];
          norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm < kEps) { norm = kEps; }
        auto scale = g_ptr[out_idx] / norm;
        for (int32_t d = 0; d < cfg_.tts_hidden_size; ++d) { w_ptr[row_offset + d] = v_ptr[row_offset + d] * scale; }
      }

      head_code_weight_.push_back(weight);
    }
  }

  Tensor makeConditionEmbeddings(const std::vector<int64_t>& text_token_ids, const std::vector<Tensor>& text_hidden_states) {
    if (text_token_ids.empty() || text_hidden_states.empty()) { return Tensor::nil(); }
    if (text_token_ids.size() != text_hidden_states.size()) {
      MLLM_ERROR("MiniCPM-o-4_5 TTS input mismatch: token count {} != hidden count {}.",
                 text_token_ids.size(), text_hidden_states.size());
      return Tensor::nil();
    }

    Tensor token_ids = Tensor::empty({1, static_cast<int32_t>(text_token_ids.size())}, kInt64, kCPU).alloc();
    for (size_t i = 0; i < text_token_ids.size(); ++i) {
      auto token_id = text_token_ids[i];
      if (token_id < 0 || token_id >= cfg_.tts_num_text_tokens) {
        MLLM_ERROR("MiniCPM-o-4_5 TTS text token id out of range: token_id={} valid=[0, {}).",
                   token_id, cfg_.tts_num_text_tokens);
        return Tensor::nil();
      }
      token_ids.at<int64_t>({0, static_cast<int32_t>(i)}) = token_id;
    }

    auto llm_embeds = emb_text_(token_ids);

    Tensor hidden_states = text_hidden_states.size() == 1 ? text_hidden_states[0] : nn::functional::concat(text_hidden_states, 1);
    auto projected_hidden = projector_semantic_(hidden_states)[0];
    if (cfg_.tts_normalize_projected_hidden) { projected_hidden = normalizeProjectedHidden(projected_hidden); }

    auto tts_embeds = llm_embeds + projected_hidden;

    Tensor text_eos = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
    text_eos.at<int64_t>({0, 0}) = cfg_.tts_text_eos_token_id;
    Tensor audio_bos = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
    audio_bos.at<int64_t>({0, 0}) = cfg_.tts_audio_bos_token_id;
    if (cfg_.tts_text_eos_token_id < 0 || cfg_.tts_text_eos_token_id >= cfg_.tts_num_text_tokens) {
      MLLM_ERROR("MiniCPM-o-4_5 TTS text_eos_token_id out of range: {} (vocab={}).",
                 cfg_.tts_text_eos_token_id, cfg_.tts_num_text_tokens);
      return Tensor::nil();
    }
    if (cfg_.tts_audio_bos_token_id < 0 || cfg_.tts_audio_bos_token_id >= cfg_.tts_num_text_tokens) {
      MLLM_ERROR("MiniCPM-o-4_5 TTS audio_bos_token_id out of range: {} (vocab={}).",
                 cfg_.tts_audio_bos_token_id, cfg_.tts_num_text_tokens);
      return Tensor::nil();
    }

    auto text_eos_embed = emb_text_(text_eos);
    auto audio_bos_embed = emb_text_(audio_bos);

    return nn::functional::concat({tts_embeds, text_eos_embed, audio_bos_embed}, 1);
  }

  MiniCPMO45TTSGenerationOutput generate(const Tensor& condition_embeds,
                                         const MiniCPMO45TTSGenerationConfig& generation_cfg = {}) {
    if (condition_embeds.isNil()) { return {}; }

    auto eos_token = cfg_.tts_num_audio_tokens - 1;

    std::vector<float> temperature = generation_cfg.temperature;
    if (temperature.empty()) { temperature.assign(cfg_.tts_num_vq, 1.0f); }
    if (temperature.size() < static_cast<size_t>(cfg_.tts_num_vq)) {
      temperature.resize(cfg_.tts_num_vq, temperature.back());
    }

    nn::StaticCache kv_cache(cfg_.tts_max_position_embeddings, cfg_.tts_num_hidden_layers,
                             cfg_.tts_num_attention_heads,  // q heads
                             cfg_.tts_num_key_value_heads,  // kv heads
                             cfg_.tts_hidden_size / cfg_.tts_num_attention_heads,
                             kFloat32,  // k dtype
                             kFloat32,  // v dtype
                             kCPU,      // device
                             false      // use fa2
    );

    Tensor generated = Tensor::zeros({1, generation_cfg.max_new_tokens, cfg_.tts_num_vq}, kInt64, kCPU);
    int32_t generated_len = 0;
    bool finished = false;
    auto condition_length = condition_embeds.shape()[1];
    std::vector<std::vector<int64_t>> generated_history(cfg_.tts_num_vq);

    for (int32_t t = 0; t < generation_cfg.max_new_tokens; ++t) {
      Tensor inputs_embeds = Tensor::nil();
      Tensor position_ids = Tensor::nil();

      if (t == 0) {
        inputs_embeds = condition_embeds;
        position_ids = Tensor::empty({1, condition_length}, kInt64, kCPU).alloc();
        for (int32_t i = 0; i < condition_length; ++i) { position_ids.at<int64_t>({0, i}) = i; }
      } else {
        for (int32_t q = 0; q < cfg_.tts_num_vq; ++q) {
          auto code_ids = generated[{kAll, {t - 1, t}, {q, q + 1}}].contiguous().view({1, 1});
          auto code_embeds = emb_code_[q](code_ids);
          if (q == 0) {
            inputs_embeds = code_embeds;
          } else {
            inputs_embeds = inputs_embeds + code_embeds;
          }
        }
        position_ids = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
        position_ids.at<int64_t>({0, 0}) = condition_length + t - 1;
      }

      auto [llm_embedding_sin, llm_embedding_cos] = llama::makeRotaryPosEmbedding(position_ids, model_.getBuffer("inv_freq"), 1.0f);
      Tensor causal_mask = Tensor::nil();
      auto* cache_ptr = static_cast<nn::AbstractStaticCache*>(&kv_cache);
      auto hidden_states = model_(inputs_embeds, llm_embedding_sin, llm_embedding_cos, causal_mask, AnyValue(cache_ptr))[0];

      auto seq_len = hidden_states.shape()[1];
      auto last_hidden = hidden_states[{kAll, {seq_len - 1, seq_len}, kAll}].contiguous();

      bool has_eos = false;
      std::vector<int64_t> step_tokens;
      step_tokens.reserve(cfg_.tts_num_vq);
      for (int32_t q = 0; q < cfg_.tts_num_vq; ++q) {
        MLLM_RT_ASSERT(q < static_cast<int32_t>(head_code_weight_.size()));
        auto logits = nn::functional::matmul(last_hidden, head_code_weight_[q], false, true)[{0, 0, kAll}].contiguous();
        auto temp = std::max(temperature[q], 1e-5f);
        logits = logits / temp;

        if (t > 0) {
          applyRepetitionPenalty(logits, generated_history[q], generation_cfg.repetition_penalty,
                                 generation_cfg.repetition_penalty_window);
          applyTopPLogits(logits, generation_cfg.top_p, 3);
          applyTopKLogits(logits, generation_cfg.top_k, 3);
        }

        if (t < generation_cfg.min_new_tokens || generation_cfg.force_no_stop) {
          if (logits.dtype() == kFloat32) {
            logits.ptr<float>()[eos_token] = -std::numeric_limits<float>::infinity();
          } else if (logits.dtype() == kFloat16) {
            logits.ptr<mllm_fp16_t>()[eos_token] = -65504.0f;
          }
        }

        bool use_sampling = generation_cfg.do_sample || generation_cfg.top_k > 0 || generation_cfg.top_p > 0.0f
                            || std::abs(temp - 1.0f) > 1e-6f;
        auto token_id = sampleFromLogits(logits, use_sampling);
        generated.at<int64_t>({0, t, q}) = token_id;
        generated_history[q].push_back(token_id);
        step_tokens.push_back(token_id);
        has_eos = has_eos || token_id == eos_token;
      }

      if (generation_cfg.step_callback) {
        auto interval = std::max(generation_cfg.debug_interval, 1);
        if (t == 0 || ((t + 1) % interval) == 0 || has_eos) {
          generation_cfg.step_callback(t + 1, step_tokens, has_eos);
        }
      }

      generated_len = t + 1;
      if (has_eos) {
        finished = true;
        break;
      }
    }

    auto out_len = generated_len;
    if (finished && out_len > 0) { out_len -= 1; }  // do not return terminal token

    Tensor out_ids = Tensor::nil();
    if (out_len > 0) { out_ids = generated[{kAll, {0, out_len}, kAll}].contiguous(); }
    return {.new_ids = out_ids, .finished = finished};
  }

 private:
  static int64_t argmax1d(const Tensor& logits) {
    auto probs = logits;
    if (probs.dtype() != kFloat32) { probs = probs.to(kFloat32); }
    auto* data = probs.ptr<float>();
    auto n = probs.shape().back();

    auto max_idx = 0;
    auto max_value = data[0];
    for (int32_t i = 1; i < n; ++i) {
      if (data[i] > max_value) {
        max_value = data[i];
        max_idx = i;
      }
    }
    return max_idx;
  }

  static int64_t sampleFromLogits(Tensor logits, bool do_sample) {
    if (logits.dtype() != kFloat32) { logits = logits.to(kFloat32); }
    if (!do_sample) { return argmax1d(logits); }

    auto probs = nn::functional::softmax(logits, -1);
    if (probs.dtype() != kFloat32) { probs = probs.to(kFloat32); }
    return categoricalSample1d(probs);
  }

  static int64_t categoricalSample1d(const Tensor& probs) {
    MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);
    auto* prob_data = probs.ptr<float>();
    auto vocab_size = probs.shape().back();

    std::vector<float> cumulative_probs(vocab_size);
    std::partial_sum(prob_data, prob_data + vocab_size, cumulative_probs.begin());

    auto total = cumulative_probs.back();
    if (total <= 0.0f) { return argmax1d(probs); }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, total);
    auto target = dist(rng);

    auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), target);
    if (it == cumulative_probs.end()) { return vocab_size - 1; }
    return static_cast<int64_t>(std::distance(cumulative_probs.begin(), it));
  }

  static void applyRepetitionPenalty(Tensor& logits, const std::vector<int64_t>& token_ids, float penalty,
                                     int32_t past_window) {
    if (penalty <= 1.0f || token_ids.empty()) { return; }
    if (logits.dtype() != kFloat32) { logits = logits.to(kFloat32); }

    auto vocab_size = logits.shape().back();
    std::unordered_map<int64_t, int32_t> frequencies;

    int32_t start = 0;
    if (past_window > 0 && static_cast<int32_t>(token_ids.size()) > past_window) {
      start = static_cast<int32_t>(token_ids.size()) - past_window;
    }
    for (int32_t i = start; i < static_cast<int32_t>(token_ids.size()); ++i) {
      auto token_id = token_ids[i];
      if (token_id < 0 || token_id >= vocab_size) { continue; }
      frequencies[token_id] += 1;
    }

    auto* logits_ptr = logits.ptr<float>();
    for (const auto& [token_id, freq] : frequencies) {
      auto alpha = std::pow(penalty, static_cast<float>(freq));
      float& value = logits_ptr[token_id];
      value = value < 0.0f ? value * alpha : value / alpha;
    }
  }

  static void applyTopKLogits(Tensor& logits, int32_t top_k, int32_t min_tokens_to_keep) {
    if (top_k <= 0) { return; }
    if (logits.dtype() != kFloat32) { logits = logits.to(kFloat32); }

    auto vocab_size = logits.shape().back();
    int32_t k = std::min(std::max(top_k, min_tokens_to_keep), vocab_size);
    if (k >= vocab_size) { return; }

    auto* logits_ptr = logits.ptr<float>();
    std::vector<int32_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&logits_ptr](int32_t lhs, int32_t rhs) { return logits_ptr[lhs] > logits_ptr[rhs]; });

    auto threshold = logits_ptr[indices[k - 1]];
    auto neg_inf = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < vocab_size; ++i) {
      if (logits_ptr[i] < threshold) { logits_ptr[i] = neg_inf; }
    }
  }

  static void applyTopPLogits(Tensor& logits, float top_p, int32_t min_tokens_to_keep) {
    if (top_p <= 0.0f || top_p >= 1.0f) { return; }
    if (logits.dtype() != kFloat32) { logits = logits.to(kFloat32); }

    auto vocab_size = logits.shape().back();
    if (vocab_size <= min_tokens_to_keep) { return; }

    auto* logits_ptr = logits.ptr<float>();
    std::vector<int32_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&logits_ptr](int32_t lhs, int32_t rhs) { return logits_ptr[lhs] > logits_ptr[rhs]; });

    auto max_logit = logits_ptr[indices[0]];
    std::vector<float> probs(vocab_size);
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
      auto prob = std::exp(logits_ptr[indices[i]] - max_logit);
      probs[i] = prob;
      sum_exp += prob;
    }
    if (sum_exp <= 0.0f) { return; }
    for (auto& p : probs) { p /= sum_exp; }

    int32_t keep = 0;
    float cumulative = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
      cumulative += probs[i];
      keep += 1;
      if (cumulative >= top_p && keep >= min_tokens_to_keep) { break; }
    }
    keep = std::max(keep, min_tokens_to_keep);
    keep = std::min(keep, vocab_size);

    auto neg_inf = -std::numeric_limits<float>::infinity();
    for (int32_t i = keep; i < vocab_size; ++i) { logits_ptr[indices[i]] = neg_inf; }
  }

  static Tensor normalizeProjectedHidden(Tensor hidden_states) {
    auto original_dtype = hidden_states.dtype();
    auto normalized = original_dtype == kFloat32 ? hidden_states.contiguous() : hidden_states.to(kFloat32).contiguous();

    auto B = normalized.shape()[0];
    auto S = normalized.shape()[1];
    auto D = normalized.shape()[2];
    auto* ptr = normalized.ptr<float>();

    constexpr float kEps = 1e-12f;
    for (int32_t b = 0; b < B; ++b) {
      for (int32_t s = 0; s < S; ++s) {
        auto base = b * S * D + s * D;
        float norm = 0.0f;
        for (int32_t d = 0; d < D; ++d) { norm += ptr[base + d] * ptr[base + d]; }
        norm = std::sqrt(norm);
        if (norm < kEps) { norm = kEps; }
        for (int32_t d = 0; d < D; ++d) { ptr[base + d] /= norm; }
      }
    }

    if (original_dtype != kFloat32) { return normalized.to(original_dtype); }
    return normalized;
  }

 private:
  MiniCPMO45Config cfg_;
  TTSProjector projector_spk_;
  TTSProjector projector_semantic_;
  std::vector<nn::Embedding> emb_code_;
  nn::Embedding emb_text_;
  std::vector<Tensor> head_code_weight_;
  llama::LlamaText model_;
};

class MiniCPMO45TextModel final : public nn::Module {
 public:
  MiniCPMO45TextModel() = default;

  MiniCPMO45TextModel(const std::string& name, const MiniCPMO45Config& cfg) : Module(name) {
    auto llm_cfg = toQwen3Config(cfg);
    decode_blocks_ = reg<nn::ModuleList<qwen3::Qwen3Decoder>>("layers", llm_cfg.num_hidden_layers, llm_cfg);
    for (auto [idx, block] : enumerate(decode_blocks_.list())) { block.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", llm_cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", llm_cfg.vocab_size, llm_cfg.hidden_size);
    registerBuffer("last_hidden_states", Tensor::nil());
  }

  Tensor embed(const Tensor& input_ids) { return embedding_(input_ids); }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto kv_cache = args[0].get<nn::StaticCache*>();

    for (auto& block : decode_blocks_.list()) {
      hidden_states = block(hidden_states, llm_embedding_sin, llm_embedding_cos, AnyValue(kv_cache))[0];
    }
    updateBuffer("last_hidden_states", hidden_states);
    hidden_states = norm_(hidden_states);
    return {hidden_states};
  }

 private:
  static qwen3::Qwen3Config toQwen3Config(const MiniCPMO45Config& cfg) {
    qwen3::Qwen3Config llm_cfg;
    llm_cfg.attention_bias = cfg.attention_bias;
    llm_cfg.hidden_size = cfg.hidden_size;
    llm_cfg.head_dim = cfg.head_dim;
    llm_cfg.intermediate_size = cfg.intermediate_size;
    llm_cfg.num_attention_heads = cfg.num_attention_heads;
    llm_cfg.num_key_value_heads = cfg.num_key_value_heads;
    llm_cfg.num_hidden_layers = cfg.num_hidden_layers;
    llm_cfg.max_position_embeddings = cfg.max_position_embeddings;
    llm_cfg.rms_norm_eps = cfg.rms_norm_eps;
    llm_cfg.vocab_size = cfg.vocab_size;
    llm_cfg.bos_token_id = cfg.bos_token_id;
    llm_cfg.eos_token_id = cfg.eos_token_id;
    llm_cfg.end_of_text_token_id = static_cast<int32_t>(cfg.eos_token_id);
    llm_cfg.rope_theta = cfg.rope_theta;
    llm_cfg.tie_word_embeddings = cfg.tie_word_embeddings;
    llm_cfg.max_cache_length = cfg.max_cache_length;
    llm_cfg.linear_impl_type = cfg.linear_impl_type;
    return llm_cfg;
  }

 private:
  nn::ModuleList<qwen3::Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  nn::Embedding embedding_;
};

class MiniCPMO45LLM final : public nn::Module {
 public:
  MiniCPMO45LLM() = default;

  MiniCPMO45LLM(const std::string& name, const MiniCPMO45Config& cfg) : nn::Module(name) {
    model_ = reg<MiniCPMO45TextModel>("model", cfg);
    lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    registerBuffer("inv_freq", qwen3::makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta));
  }

  Tensor embed(const Tensor& input_ids) { return model_.embedding_(input_ids); }

  Tensor logits(const Tensor& hidden_states) { return lm_head_(hidden_states); }

  Tensor hiddenStates(Tensor& input_embeddings, Tensor& llm_embedding_sin, Tensor& llm_embedding_cos, nn::StaticCache* kv_cache) {
    return model_(input_embeddings, llm_embedding_sin, llm_embedding_cos, AnyValue(kv_cache))[0];
  }

 public:
  MiniCPMO45TextModel model_;

 private:
  nn::Linear lm_head_;
};

class MiniCPMO45ForCausalLM : public models::ARGeneration {
 public:
  struct TextGenerationWithHiddenOutput {
    std::vector<int64_t> generated_tokens;
    std::vector<int64_t> aligned_tokens;
    std::vector<Tensor> aligned_hidden_states;
    bool finished = false;
  };

  explicit MiniCPMO45ForCausalLM(const MiniCPMO45Config& config)
      : config_(config),
        legacy_config_(createLegacyConfig(config)),
        llm_("llm", config),
        vpm_("vpm", legacy_config_),
        resampler_("resampler", config.query_num, config.hidden_size, config.num_attention_heads, config.vision_hidden_size),
        apm_("apm", legacy_config_),
        audio_projection_layer_("audio_projection_layer", config.audio_hidden_size, config.hidden_size, config.hidden_size),
        audio_avg_pooler_("audio_avg_pooler", config.audio_pool_step, config.audio_pool_step),
        tts_("tts", config),
        kv_cache_(config.max_cache_length, config.num_hidden_layers,
                  config.num_attention_heads,  // q heads
                  config.num_key_value_heads,  // kv heads
                  config.head_dim,             // kv dim
                  kFloat32,                    // k dtype
                  kFloat32,                    // v dtype
                  kCPU,                        // device
                  false                        // use fa2
        ) {
    eos_token_id_ = static_cast<int32_t>(config.eos_token_id);
    max_length_ = config.max_cache_length;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& inputs, const ARGenerationArgs& args) override {
    Tensor input_ids = Tensor::nil();
    if (inputs.count("input_ids")) {
      input_ids = inputs.at("input_ids");
    } else if (inputs.count("sequence")) {
      input_ids = inputs.at("sequence");
    } else {
      MLLM_ERROR("No input_ids or sequence found in MiniCPM-o-4_5 forward input.");
      return {};
    }

    auto input_embeddings = llm_.embed(input_ids);

    Tensor prev_position_ids = inputs.count("position_ids") ? inputs.at("position_ids") : Tensor::nil();

    // Prefill-only multimodal embedding injection.
    if (prev_position_ids.isNil()) {
      auto pixel_values = inputs.count("pixel_values") ? inputs.at("pixel_values") : Tensor::nil();
      auto tgt_sizes = inputs.count("tgt_sizes") ? inputs.at("tgt_sizes") : Tensor::nil();
      auto image_bounds = inputs.count("image_bounds") ? inputs.at("image_bounds") : Tensor::nil();

      if (!pixel_values.isNil() && !tgt_sizes.isNil() && !image_bounds.isNil()) {
        auto vision_outputs = vpm_(pixel_values, tgt_sizes)[0];
        auto vision_embeddings = resampler_(vision_outputs, tgt_sizes)[0];
        input_embeddings = mergeVisionTextEmbeddings(input_embeddings, vision_embeddings, image_bounds);
      }

      auto audio_features = inputs.count("audio_features") ? inputs.at("audio_features") : Tensor::nil();
      auto audio_bounds = inputs.count("audio_bounds") ? inputs.at("audio_bounds") : Tensor::nil();

      if (!audio_features.isNil() && !audio_bounds.isNil()) {
        auto audio_embeddings = encodeAudio(audio_features);
        input_embeddings = mergeAudioTextEmbeddings(input_embeddings, audio_embeddings, audio_bounds);
      }
    }

    Tensor position_ids = makePositionIds(input_embeddings.shape()[1], prev_position_ids);

    auto [llm_embedding_sin, llm_embedding_cos] = qwen3::makeRotaryPosEmbedding(position_ids, llm_.getBuffer("inv_freq"), 1.0f);

    auto hidden_states = llm_.hiddenStates(input_embeddings, llm_embedding_sin, llm_embedding_cos, &kv_cache_);
    auto seq_len = hidden_states.shape()[1];
    auto last_hidden = hidden_states[{kAll, {seq_len - 1, seq_len}, kAll}].contiguous();
    auto logits = llm_.logits(last_hidden);

    return {
        {"sequence", logits},
        {"position_ids", position_ids},
        {"last_hidden", last_hidden},
    };
  }

  Tensor encodeAudio(const Tensor& audio_features) {
    // 1) Whisper encoder.
    auto audio_states = apm_(audio_features)[0];

    // 2) Project to the LLM hidden space.
    auto audio_embeds = audio_projection_layer_(audio_states)[0];

    // 3) Temporal pooling.
    audio_embeds = audio_embeds.transpose(1, 2);
    audio_embeds = audio_avg_pooler_(audio_embeds)[0];
    audio_embeds = audio_embeds.transpose(1, 2);
    return audio_embeds;
  }

  TextGenerationWithHiddenOutput generateTextWithHidden(const ARGenerationOutputPast& initial_inputs, int32_t max_new_tokens,
                                                        const std::vector<int64_t>& stop_token_ids, bool do_sample = false,
                                                        float temperature = 1.0f, int32_t top_k = 0, float top_p = 0.0f,
                                                        const std::function<void(int32_t step, int64_t token_id)>& step_callback =
                                                            nullptr) {
    TextGenerationWithHiddenOutput result;

    auto current_input = initial_inputs;
    bool has_previous_generated = false;
    int64_t previous_generated_token = 0;

    for (int32_t i = 0; i < max_new_tokens; ++i) {
      auto output = forward(current_input, {});

      if (has_previous_generated && output.count("last_hidden")) {
        result.aligned_tokens.push_back(previous_generated_token);
        result.aligned_hidden_states.push_back(output.at("last_hidden").contiguous().clone());
      }

      Tensor logits = output.at("sequence");
      int64_t next_token_id = 0;
      if (do_sample || temperature != 1.0f || top_k > 0 || top_p > 0.0f) {
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
      result.generated_tokens.push_back(next_token_id);
      if (step_callback) { step_callback(i + 1, next_token_id); }

      if (isStopToken(next_token_id, stop_token_ids)) {
        result.finished = true;
        break;
      }

      current_input = std::move(output);
      current_input["sequence"] = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
      current_input["sequence"].at<int64_t>({0, 0}) = next_token_id;

      previous_generated_token = next_token_id;
      has_previous_generated = true;
    }

    if (!result.finished && has_previous_generated
        && result.aligned_tokens.size() + 1 == result.generated_tokens.size()) {
      auto probe_output = forward(current_input, {});
      if (probe_output.count("last_hidden")) {
        result.aligned_tokens.push_back(previous_generated_token);
        result.aligned_hidden_states.push_back(probe_output.at("last_hidden").contiguous().clone());
      }
    }

    return result;
  }

 public:
  MiniCPMO45Config config_;
  minicpmo::MiniCPMOConfig legacy_config_;

  MiniCPMO45LLM llm_;
  minicpmo::SiglipVisionModel vpm_;
  minicpmo::Resampler resampler_;

  minicpmo::WhisperEncoder apm_;
  AudioProjectionLayer audio_projection_layer_;
  AudioAvgPooler audio_avg_pooler_;
  MiniCPMO45TTS tts_;

 private:
  template <typename DType>
  static void copyEmbeddingVector(Tensor& dst, const Tensor& src, int32_t dst_batch, int32_t dst_pos, int32_t src_batch,
                                  int32_t src_pos, int32_t hidden_size) {
    auto* dst_ptr = dst.offsettedPtr<DType>({dst_batch, dst_pos, 0});
    auto* src_ptr = src.coffsettedPtr<DType>({src_batch, src_pos, 0});
    std::memcpy(dst_ptr, src_ptr, hidden_size * sizeof(DType));
  }

  static Tensor mergeVisionTextEmbeddings(Tensor& text_embeddings, Tensor& vision_embeddings, const Tensor& image_bounds) {
    auto batch_size = text_embeddings.shape()[0];
    auto hidden_size = text_embeddings.shape()[2];
    auto vision_seq_len = vision_embeddings.shape()[1];
    auto num_bounds = std::min(image_bounds.shape()[0], vision_embeddings.shape()[0]);

    if (vision_embeddings.shape()[0] != image_bounds.shape()[0]) {
      MLLM_WARN("MiniCPM-o-4_5 vision bound count ({}) != embedding group count ({}). Using min={}.",
                image_bounds.shape()[0], vision_embeddings.shape()[0], num_bounds);
    }

    if (vision_embeddings.dtype() != text_embeddings.dtype()) { vision_embeddings = vision_embeddings.to(text_embeddings.dtype()); }

    for (int32_t b = 0; b < batch_size; ++b) {
      for (int32_t bound_idx = 0; bound_idx < num_bounds; ++bound_idx) {
        int32_t vision_idx = 0;
        auto start_pos = image_bounds.constAt<int32_t>({bound_idx, 0}) + 1;
        auto end_pos = image_bounds.constAt<int32_t>({bound_idx, 1}) - 1;

        for (int32_t pos = start_pos; pos <= end_pos && vision_idx < vision_seq_len; ++pos, ++vision_idx) {
          if (text_embeddings.dtype() == kFloat32) {
            copyEmbeddingVector<float>(text_embeddings, vision_embeddings, b, pos, bound_idx, vision_idx, hidden_size);
          } else if (text_embeddings.dtype() == kFloat16) {
            copyEmbeddingVector<mllm_fp16_t>(text_embeddings, vision_embeddings, b, pos, bound_idx, vision_idx, hidden_size);
          } else {
            MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported text embedding dtype in MiniCPM-o-4_5 vision merge.");
          }
        }
      }
    }
    return text_embeddings;
  }

  static Tensor mergeAudioTextEmbeddings(Tensor& text_embeddings, Tensor& audio_embeddings, const Tensor& audio_bounds) {
    auto batch_size = text_embeddings.shape()[0];
    auto hidden_size = text_embeddings.shape()[2];
    auto audio_seq_len = audio_embeddings.shape()[1];
    auto num_bounds = std::min(audio_bounds.shape()[0], audio_embeddings.shape()[0]);

    if (audio_embeddings.shape()[0] != audio_bounds.shape()[0]) {
      MLLM_WARN("MiniCPM-o-4_5 audio bound count ({}) != embedding group count ({}). Using min={}.",
                audio_bounds.shape()[0], audio_embeddings.shape()[0], num_bounds);
    }

    if (audio_embeddings.dtype() != text_embeddings.dtype()) { audio_embeddings = audio_embeddings.to(text_embeddings.dtype()); }

    for (int32_t b = 0; b < batch_size; ++b) {
      for (int32_t bound_idx = 0; bound_idx < num_bounds; ++bound_idx) {
        int32_t audio_idx = 0;
        auto start_pos = audio_bounds.constAt<int32_t>({bound_idx, 0});
        auto end_pos = audio_bounds.constAt<int32_t>({bound_idx, 1}) - 1;

        for (int32_t pos = start_pos; pos <= end_pos && audio_idx < audio_seq_len; ++pos, ++audio_idx) {
          if (text_embeddings.dtype() == kFloat32) {
            copyEmbeddingVector<float>(text_embeddings, audio_embeddings, b, pos, bound_idx, audio_idx, hidden_size);
          } else if (text_embeddings.dtype() == kFloat16) {
            copyEmbeddingVector<mllm_fp16_t>(text_embeddings, audio_embeddings, b, pos, bound_idx, audio_idx, hidden_size);
          } else {
            MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported text embedding dtype in MiniCPM-o-4_5 audio merge.");
          }
        }
      }
    }
    return text_embeddings;
  }

  Tensor makePositionIds(int32_t seq_len, const Tensor& prev_position_ids) {
    Tensor position_ids = Tensor::empty({1, seq_len}, kInt64).alloc();
    if (!prev_position_ids.isNil()) {
      auto last_pos = *prev_position_ids.coffsettedPtr<int64_t>({0, prev_position_ids.shape()[1] - 1});
      for (int32_t i = 0; i < seq_len; ++i) { position_ids.at<int64_t>({0, i}) = last_pos + i + 1; }
      return position_ids;
    }

    auto last_seen_tokens = kv_cache_.getCurrentSeqCnt(0);
    for (int32_t i = 0; i < seq_len; ++i) { position_ids.at<int64_t>({0, i}) = last_seen_tokens + i; }
    return position_ids;
  }

  static minicpmo::MiniCPMOConfig createLegacyConfig(const MiniCPMO45Config& config) {
    minicpmo::MiniCPMOConfig legacy;
    legacy.vision_hidden_size = config.vision_hidden_size;
    legacy.vision_intermediate_size = config.vision_intermediate_size;
    legacy.vision_num_hidden_layers = config.vision_num_hidden_layers;
    legacy.vision_num_attention_heads = config.vision_num_attention_heads;
    legacy.vision_num_channels = config.vision_num_channels;
    legacy.vision_image_size = config.vision_image_size;
    legacy.vision_patch_size = config.vision_patch_size;

    legacy.hidden_size = config.hidden_size;
    legacy.intermediate_size = config.intermediate_size;
    legacy.num_attention_heads = config.num_attention_heads;
    legacy.num_key_value_heads = config.num_key_value_heads;
    legacy.num_hidden_layers = config.num_hidden_layers;
    legacy.max_position_embeddings = config.max_position_embeddings;
    legacy.rms_norm_eps = config.rms_norm_eps;
    legacy.vocab_size = config.vocab_size;

    legacy.query_num = config.query_num;

    legacy.audio_hidden_size = config.audio_hidden_size;
    legacy.audio_num_hidden_layers = config.audio_num_hidden_layers;
    legacy.audio_num_attention_heads = config.audio_num_attention_heads;
    legacy.audio_max_position_embeddings = config.audio_max_position_embeddings;
    legacy.audio_chunk_length = config.audio_chunk_length;
    legacy.audio_pool_step = config.audio_pool_step;

    legacy.max_cache_length = config.max_cache_length;
    legacy.eos_token_id = config.eos_token_id;
    legacy.bos_token_id = config.bos_token_id;
    legacy.rope_theta = config.rope_theta;
    legacy.tie_word_embeddings = config.tie_word_embeddings;

    legacy.linear_impl_type = config.linear_impl_type;
    return legacy;
  }

  static bool isStopToken(int64_t token_id, const std::vector<int64_t>& stop_token_ids) {
    for (auto id : stop_token_ids) {
      if (token_id == id) { return true; }
    }
    return false;
  }

 private:
  nn::StaticCache kv_cache_;
};

}  // namespace mllm::models::minicpm_o45
