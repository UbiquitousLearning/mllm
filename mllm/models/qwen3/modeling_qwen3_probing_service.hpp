// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <span>
#include <memory>
#include <ranges>
#include <algorithm>
#include <filesystem>
#include <map>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <set>

#include <nlohmann/json.hpp>

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

// Service related.
#include "mllm/engine/service/Session.hpp"
#include "mllm/engine/prefix_cache/Cache.hpp"

namespace mllm::models::qwen3_probing {

using namespace mllm;
using namespace mllm::nn;
using namespace mllm::models::qwen3;

struct ProbingArgs {
  bool enable_prefill_check = false;
  float prefill_stop_threshold = 0.7f;
  std::vector<int> default_prefill_layers;

  bool enable_decode_check = false;
  float decode_stop_threshold = 0.8f;
  float pos_threshold = 0.9f;
};

struct ProbingContext {
  std::map<int, Tensor> mlp_outputs;
  bool collecting = false;
  bool save_last_token_only = false;
  std::set<int> target_layers;

  void reset() {
    mlp_outputs.clear();
    collecting = false;
    save_last_token_only = false;
    target_layers.clear();
  }

  void soft_reset() {
    collecting = false;
    save_last_token_only = false;
    target_layers.clear();
  }
};

// RoPE
inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                   float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;
  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }
  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;
        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }
  return {sin_emb, cos_emb};
}

// Linear, Scaler, PCA
class ProbeClassifier : public Module {
  Linear linear_;

  bool use_scaler_ = false;
  Param scaler_mean_;
  Param scaler_scale_;

  bool use_pca_ = false;
  Param pca_components_;  // [hidden_dim, pca_dim]

 public:
  ProbeClassifier() = default;

  ProbeClassifier(const std::string& name, int hidden_dim, int linear_in_dim, bool use_scaler, bool use_pca,
                  std::string linear_name, std::string scaler_prefix, std::string pca_name)
      : Module(name), use_scaler_(use_scaler), use_pca_(use_pca) {
    if (use_scaler_) {
      scaler_mean_ = reg<Param>("scaler_mean", scaler_prefix + "_mean.weight", Tensor::shape_t({1, 1, hidden_dim}));
      scaler_scale_ = reg<Param>("scaler_scale", scaler_prefix + "_scale.weight", Tensor::shape_t({1, 1, hidden_dim}));
    }

    if (use_pca_) {
      pca_components_ = reg<Param>("pca_components", pca_name + ".weight", Tensor::shape_t({linear_in_dim, hidden_dim}));
    }
    linear_ = reg<Linear>(linear_name, linear_in_dim, 1, true, mllm::aops::LinearImplTypes::kDefault);
  }

  virtual float predict(Tensor& hidden_emb) {
    Tensor x = hidden_emb;

    if (use_scaler_) {
      x = x - scaler_mean_.weight();
      x = x / scaler_scale_.weight();
    }
    if (use_pca_) {
      // hidden_emb: [1, 1, hidden_dim]
      // pca_components_.weight(): [linear_in_dim, hidden_dim]
      // transpose(0, 1): [hidden_dim, linear_in_dim]
      // matmul: [1, 1, hidden_dim] * [hidden_dim, linear_in_dim] -> [1, 1, linear_in_dim]
      x = mllm::nn::functional::matmul(x, pca_components_.weight().transpose(0, 1));
    }

    auto logits = linear_(x);

    float val = 0.0f;
    if (logits.dtype() == mllm::kFloat32) {
      val = logits.ptr<float>()[0];
    } 

    return 1.0f / (1.0f + std::exp(-val));
  }
};

// MODEL
class Qwen3ProbingMLP final : public Module {
  Linear gate_proj_, up_proj_, down_proj_;
  SiLU silu_;

 public:
  Qwen3ProbingMLP() = default;
  Qwen3ProbingMLP(const std::string& name, const Qwen3Config& cfg) : Module(name) {
    gate_proj_ = reg<Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    silu_ = reg<SiLU>("act");
    up_proj_ = reg<Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }
};

class Qwen3ProbingAttention final : public Module {
  Linear q_proj_, k_proj_, v_proj_, o_proj_;
  RMSNorm rms_norm_q_, rms_norm_k_;
  RoPE q_rope_, k_rope_;
  RadixAttn attn_;
  int hidden_size_, head_dim_, num_attention_heads_, num_key_value_heads_;
  int layer_idx_;

 public:
  friend class Qwen3ProbingText;
  friend class Qwen3ProbingDecoder;

  Qwen3ProbingAttention() = default;
  Qwen3ProbingAttention(const std::string& name, const Qwen3Config& cfg) : Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    q_proj_ = reg<Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ = reg<Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    v_proj_ = reg<Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    o_proj_ = reg<Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);
    rms_norm_q_ = reg<RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<RMSNorm>("k_norm", cfg.rms_norm_eps);
    q_rope_ = reg<RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings, aops::RoPEOpOptionsInputType::kBSHD);
    k_rope_ = reg<RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings, aops::RoPEOpOptionsInputType::kBSHD);
    attn_ = reg<RadixAttn>("attn", num_attention_heads_, num_key_value_heads_);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto k_cache_addr = args[0].get<std::vector<std::vector<prefix_cache::vp_addr_t>>*>();
    auto v_cache_addr = args[1].get<std::vector<std::vector<prefix_cache::vp_addr_t>>*>();
    auto prefix_cache_context = args[2].get<prefix_cache::Cache*>();

    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    std::vector<prefix_cache::vp_addr_t> k_addr_wait_for_promote, v_addr_wait_for_promote;
    for (int s_idx = 0; s_idx < S; ++s_idx) {
      k_addr_wait_for_promote.push_back(prefix_cache_context->alloc(kCPU));
      v_addr_wait_for_promote.push_back(prefix_cache_context->alloc(kCPU));
    }
    std::vector<char*> k_phy_addr_wait_for_promote, v_phy_addr_wait_for_promote;
    for (int s_idx = 0; s_idx < S; ++s_idx) {
      k_phy_addr_wait_for_promote.push_back(prefix_cache_context->physicalAddr(k_addr_wait_for_promote[s_idx]));
      v_phy_addr_wait_for_promote.push_back(prefix_cache_context->physicalAddr(v_addr_wait_for_promote[s_idx]));
    }
    auto k_wait_for_promote = Tensor::refVectorData(k_phy_addr_wait_for_promote, {S}, kInt64, kCPU);
    auto v_wait_for_promote = Tensor::refVectorData(v_phy_addr_wait_for_promote, {S}, kInt64, kCPU);

    nn::functional::scatter2Shards(key_states, k_wait_for_promote, 1);
    nn::functional::scatter2Shards(value_states, v_wait_for_promote, 1);

    {
      auto& dst = (*k_cache_addr)[layer_idx_];
      dst.insert(dst.end(), k_addr_wait_for_promote.begin(), k_addr_wait_for_promote.end());
    }
    {
      auto& dst = (*v_cache_addr)[layer_idx_];
      dst.insert(dst.end(), v_addr_wait_for_promote.begin(), v_addr_wait_for_promote.end());
    }

    std::vector<char*> k_phy_cache_indicies, v_phy_cache_indicies;
    int32_t kv_cache_len = (*k_cache_addr)[layer_idx_].size();
    k_phy_cache_indicies.reserve(kv_cache_len);
    v_phy_cache_indicies.reserve(kv_cache_len);
    for (int i = 0; i < kv_cache_len; ++i) {
      k_phy_cache_indicies.push_back(prefix_cache_context->physicalAddr((*k_cache_addr)[layer_idx_][i]));
      v_phy_cache_indicies.push_back(prefix_cache_context->physicalAddr((*v_cache_addr)[layer_idx_][i]));
    }
    auto k_cache = Tensor::refVectorData(k_phy_cache_indicies, {kv_cache_len}, kInt64, kCPU);
    auto v_cache = Tensor::refVectorData(v_phy_cache_indicies, {kv_cache_len}, kInt64, kCPU);

    auto output = attn_(query_states, k_cache, v_cache);
    output = output.view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    return {output};
  }
};

class Qwen3ProbingDecoder final : public Module {
 public:
  Qwen3ProbingAttention self_attn_;
  Qwen3ProbingMLP mlp_;
  RMSNorm input_layer_norm_, post_attention_layer_norm_;
  int layer_idx_;

  Qwen3ProbingDecoder() = default;
  Qwen3ProbingDecoder(const std::string& name, const Qwen3Config& cfg) : Module(name) {
    self_attn_ = reg<Qwen3ProbingAttention>("self_attn", cfg);
    mlp_ = reg<Qwen3ProbingMLP>("mlp", cfg);
    input_layer_norm_ = reg<RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& k_cache_addr = args[0];
    auto& v_cache_addr = args[1];
    auto& prefix_cache_context = args[2];

    ProbingContext* probe_ctx = nullptr;
    if (args.size() > 3) probe_ctx = args[3].get<ProbingContext*>();

    auto x = input_layer_norm_(inputs[0]);
    x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, k_cache_addr, v_cache_addr, prefix_cache_context)[0];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);

    auto mlp_out = mlp_(x)[0];

    // Probe
    if (probe_ctx && probe_ctx->collecting) {
      bool layer_needed = false;
      if (probe_ctx->target_layers.empty())
        layer_needed = true;
      else if (probe_ctx->target_layers.count(layer_idx_))
        layer_needed = true;

      if (layer_needed) {
        int batch = mlp_out.shape()[0];
        int seq_len = mlp_out.shape()[1];
        int hidden_dim = mlp_out.shape()[2];

        Tensor* dest_ptr = nullptr;
        bool need_alloc = true;
        if (probe_ctx->mlp_outputs.count(layer_idx_)) {
          auto& t = probe_ctx->mlp_outputs[layer_idx_];
          if (t.shape().size() == 3 && t.shape()[0] == batch && t.shape()[1] == 1 && t.shape()[2] == hidden_dim
              && t.dtype() == mlp_out.dtype()) {
            dest_ptr = &t;
            need_alloc = false;
          }
        }

        if (need_alloc) {
          probe_ctx->mlp_outputs[layer_idx_] = Tensor::empty({batch, 1, hidden_dim}, mlp_out.dtype(), kCPU);
          probe_ctx->mlp_outputs[layer_idx_].alloc();
          dest_ptr = &probe_ctx->mlp_outputs[layer_idx_];
        }

        int token_offset = probe_ctx->save_last_token_only ? (seq_len - 1) : 0;

        size_t dtype_size = (mlp_out.dtype() == mllm::kFloat32) ? 4 : 2;
        char* src_base_ptr = (char*)mlp_out.ptr<float>();
        size_t byte_offset = (size_t)token_offset * hidden_dim * dtype_size;

        if (src_base_ptr && dest_ptr->ptr<float>()) {
          std::memcpy(dest_ptr->ptr<float>(), src_base_ptr + byte_offset, hidden_dim * dtype_size);
        }
      }
    }

    x = mlp_out + tmp;
    return {x};
  }
};

class Qwen3ProbingText final : public Module {
  ModuleList<Qwen3ProbingDecoder> decode_blocks_;
  RMSNorm norm_;
  Embedding embedding_;

 public:
  Qwen3ProbingText() = default;
  Qwen3ProbingText(const std::string& name, const Qwen3Config& cfg) : Module(name) {
    decode_blocks_ = reg<ModuleList<Qwen3ProbingDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) {
      b.self_attn_.layer_idx_ = idx;
      b.layer_idx_ = idx;
    }
    norm_ = reg<RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();
    auto x = embedding_(inputs[0]);
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];

    for (auto& block : blocks) {
      x = block(x, llm_embedding_sin, llm_embedding_cos, args[0], args[1], args[2], args.size() > 3 ? args[3] : AnyValue())[0];
    }
    x = norm_(x);
    return {x};
  }
};

class Qwen3ProbingForCausalLM : public ARGeneration, public Module {
 public:
  explicit Qwen3ProbingForCausalLM(const Qwen3Config& cfg) : cfg(cfg) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;
    llm = reg<Qwen3ProbingText>("model", cfg);
    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<Linear>("lm_head_out", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }
    auto inv = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  // load probes from directory
  void loadProbesFromDirectory(const std::string& dir_path, const ProbingArgs& args) {
    namespace fs = std::filesystem;
    if (!fs::exists(dir_path)) {
      std::cerr << "Probe dir not found: " << dir_path << std::endl;
      return;
    }

    for (const auto& entry : fs::directory_iterator(dir_path)) {
      std::string fn = entry.path().filename().string();
      std::string path = entry.path().string();
      if (fn.find(".mllm") == std::string::npos) continue;

      std::shared_ptr<mllm::ParameterFile> params;
      try {
        params = mllm::load(path, mllm::ModelFileVersion::kV2);
      } catch (const std::exception& e) {
        std::cerr << "Failed to open " << fn << ": " << e.what() << std::endl;
        continue;
      }

      std::string detect_linear_name = "classifier";
      std::string detect_scaler_prefix = "scaler";
      std::string detect_pca_name = "pca_components";

      bool has_scaler = false;
      bool has_pca = false;
      int linear_in_dim = cfg.hidden_size;

      for (auto& [key, tensor] : *params) {
        if (key.find("linear.weight") != std::string::npos) {
          detect_linear_name = "linear";
          if (tensor.shape().size() > 1) linear_in_dim = tensor.shape()[1];
        } else if (key.find("classifier.weight") != std::string::npos) {
          detect_linear_name = "classifier";
          if (tensor.shape().size() > 1) linear_in_dim = tensor.shape()[1];
        }

        if (key.find("scaler_mean") != std::string::npos) {
          has_scaler = true;
          detect_scaler_prefix = "scaler";
        }

        if (key.find("pca_proj") != std::string::npos) {
          has_pca = true;
          detect_pca_name = "pca_proj";
        } else if (key.find("pca_components") != std::string::npos) {
          has_pca = true;
          detect_pca_name = "pca_components";
        }
      }

      bool is_prefill = (fn.find("prefill") != std::string::npos);
      bool is_pos = (fn.find("pos_probe") != std::string::npos);

      int parsed_layer = -1;
      size_t layer_pos = fn.find("layer-");
      if (layer_pos != std::string::npos) {
        try {
          size_t num_start = layer_pos + 6;
          size_t num_end = fn.find_first_not_of("0123456789", num_start);
          parsed_layer = std::stoi(fn.substr(num_start, num_end - num_start));
        } catch (...) {}
      }

      bool use_scaler = has_scaler;
      bool use_pca = has_pca;

      if (!use_pca && linear_in_dim != cfg.hidden_size) {
        // If linear_in_dim differs from hidden_size, PCA must be used
      }

      std::cout << "  -> Loading " << fn << " [S:" << (use_scaler ? "ON" : "OFF") << ", P:" << (use_pca ? "ON" : "OFF")
                << ", Dim:" << linear_in_dim << ", Layer:" << parsed_layer << "]" << std::endl;

      auto probe = std::make_shared<ProbeClassifier>("", cfg.hidden_size, linear_in_dim, use_scaler, use_pca,
                                                     detect_linear_name, detect_scaler_prefix, detect_pca_name);
      try {
        probe->load(params);
      } catch (const std::exception& e) {
        std::cerr << "Error loading weights for " << fn << ": " << e.what() << std::endl;
        continue;
      }

      if (is_pos) {
        pos_probe = probe;
        if (parsed_layer != -1) pos_probe_layer_idx = parsed_layer;
      } else {
        if (parsed_layer == -1) continue;
        if (is_prefill)
          prefill_probes[parsed_layer].push_back(probe);
        else
          decode_probes[parsed_layer].push_back(probe);
      }
    }
    std::cout << "Loaded Summary: Prefill(" << prefill_probes.size() << "), Decode(" << decode_probes.size() << "), Pos("
              << (pos_probe ? "Yes@L" + std::to_string(pos_probe_layer_idx) : "No") << ")" << std::endl;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    // Standard forward pass (same as before)
    auto sequence = input.at("sequence");
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];
    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }
    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);
    std::vector<AnyValue> forward_args = {args.at("k_cache_addrs"), args.at("v_cache_addrs"), args.at("prefix_cache_context")};
    if (args.count("probing_context")) forward_args.push_back(args.at("probing_context"));

    sequence = llm(sequence, llm_embedding_sin, llm_embedding_cos, forward_args)[0];
    {
      auto S = sequence.shape()[1];
      sequence = sequence[{kAll, {S - 1}, kAll}];
    }
    if (tie_word_embeddings_) { sequence = lm_head_(sequence); }
    return {{"sequence", sequence}, {"position_ids", position_ids}};
  }

  const Qwen3Config cfg;
  std::map<int, std::vector<std::shared_ptr<ProbeClassifier>>> prefill_probes;
  std::map<int, std::vector<std::shared_ptr<ProbeClassifier>>> decode_probes;
  std::shared_ptr<ProbeClassifier> pos_probe;
  int pos_probe_layer_idx = -1;

  // Public exposure for collector
  struct ProbeResult {
    float score;
    int layer;
    std::string phase;
    std::string type = "hallucination";  // "hallucination" or "pos_check"
    bool is_key_predicted = false;
    int token_idx = -1;
    int token_id = -1;
  };
  std::vector<ProbeResult> last_probe_results_;
  void clearProbeResults() { last_probe_results_.clear(); }

 private:
  Qwen3ProbingText llm;
  Linear lm_head_;
  bool tie_word_embeddings_;
};

// Session
class Qwen3ProbingSession final : public ::mllm::service::Session {
 public:
  Qwen3ProbingSession() = default;

  void setProbingArgs(const ProbingArgs& args) { probing_args_ = args; }
  void loadProbes(const std::string& path, const ProbingArgs& args) { model_->loadProbesFromDirectory(path, args); }

  std::vector<Qwen3ProbingForCausalLM::ProbeResult> getLastProbeResults() { return model_->last_probe_results_; }
  void clearLastProbeResults() { model_->clearProbeResults(); }

  std::size_t findThinkStartToken(const std::vector<int64_t>& output_ids) {
    auto it = std::find(output_ids.begin(), output_ids.end(), model_->cfg.thinking_start_token_id);
    return std::distance(output_ids.begin(), it);
  }

  void streamGenerate(const nlohmann::json& request,
                      const std::function<void(const nlohmann::json&, bool)>& callback) override {
    mllm::cpu::wakeupHpcThreadPool();
    auto messages = request["messages"];

    // 简短指令
    std::string concise_instruction = " Please answer in a single, complete sentence. Keep it concise.";

    bool has_system = false;
    if (!messages.empty() && messages[0].value("role", "") == "system") {
      std::string current_content = messages[0].value("content", "");
      messages[0]["content"] = current_content + concise_instruction;
      has_system = true;
    }

    if (!has_system) {
      nlohmann::json sys_msg;
      sys_msg["role"] = "system";
      sys_msg["content"] = "You are a helpful assistant." + concise_instruction;
      messages.insert(messages.begin(), sys_msg);
    }
    auto inputs = applyChatTemplate(messages, {}, true, request.value("enable_thinking", false));
    auto full_seq_idx = tokenizer_->convert2Ids(tokenizer_->tokenize(inputs)).toVector<int64_t>();

    ARGenerationArgs args;
    ARGenerationOutputPast input;
    auto prefix_cache_result = cache_->find(full_seq_idx);
    std::span<int64_t> reduced_seq_idx(full_seq_idx.data() + prefix_cache_result.matched_length,
                                       full_seq_idx.size() - prefix_cache_result.matched_length);
    std::vector<int64_t> position_ids;
    {
      auto start = prefix_cache_result.matched_length;
      auto end = full_seq_idx.size();
      position_ids.reserve(end - start);
      std::ranges::copy(std::views::iota(static_cast<int64_t>(start), static_cast<int64_t>(end)),
                        std::back_inserter(position_ids));
    }
    input["sequence"] = Tensor::fromVector(reduced_seq_idx, {1, (int32_t)reduced_seq_idx.size()}, kInt64, kCPU);
    input["position_ids"] = Tensor::fromVector(position_ids, {1, (int32_t)position_ids.size()}, kInt64, kCPU);
    k_cache_addrs_ = prefix_cache_result.k_cache_addresses;
    v_cache_addrs_ = prefix_cache_result.v_cache_addresses;

    ProbingContext probe_ctx;
    args["k_cache_addrs"] = &k_cache_addrs_;
    args["v_cache_addrs"] = &v_cache_addrs_;
    args["prefix_cache_context"] = cache_.get();
    args["probing_context"] = &probe_ctx;
    args["temperature"] = request.value("temperature", 1.0f);
    args["top_k"] = request.value("top_k", 0);
    args["top_p"] = request.value("top_p", 0.0f);
    auto max_length = request.value("max_length", 1024);
    args["max_length"] = max_length;
    args["do_sample"] = request.value("do_sample", false);

    bool stop_generating = false;

    // get Prefill Layers
    std::vector<int> current_prefill_layers;
    if (request.contains("prefill_layers") && request["prefill_layers"].is_array()) {
      current_prefill_layers = request["prefill_layers"].get<std::vector<int>>();
    } else {
      current_prefill_layers = probing_args_.default_prefill_layers;
    }

    if (probing_args_.enable_prefill_check && !current_prefill_layers.empty()) {
      probe_ctx.reset();
      probe_ctx.collecting = true;
      probe_ctx.save_last_token_only = true;
      for (int l : current_prefill_layers) probe_ctx.target_layers.insert(l);
    }

    struct CandidateKey {
      int token_idx;
      int token_id;
      float score;
      std::map<int, Tensor> activations;
    };
    std::shared_ptr<CandidateKey> candidate_key = nullptr;
    int debounce_counter = 0;
    bool has_confirmed_key_in_decode = false;

    int64_t package_cnt = 0;
    std::string accumulated_output = "";
    auto wrapped_callback = [this, &max_length, &request, &full_seq_idx, &package_cnt, &callback, &probe_ctx, &stop_generating,
                             &candidate_key, &debounce_counter, &has_confirmed_key_in_decode,
                             &accumulated_output](int64_t idx) {
      if (stop_generating) return;

      // Calculate token string early for punctuation check
      std::string current_token_str = preprocessor::wideString2Utf8String(tokenizer_->detokenize(idx));

      // 0. Accumulate output (Wait for safety check)
      // Do not append EOS token to the buffer
      if (idx != model_->cfg.eos_token_id) { accumulated_output += current_token_str; }

      auto should_skip_token = [](std::string s) -> bool {
        // 1. Trim leading/trailing whitespace (include common whitespace chars)
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return true;  // All spaces/empty
        size_t end = s.find_last_not_of(" \t\n\r");
        std::string core = s.substr(start, end - start + 1);

        // 2. Check if Punctuation (all chars are punct)
        bool all_punct = true;
        for (unsigned char c : core) {
          // If we find an alphanumeric char, it's NOT (just) punctuation
          if (isalnum(c)) {
            all_punct = false;
            break;
          }
          // If high-bit (likely UTF-8 chinese/emoji), treat as Non-Punctuation for now (unless specific symbol list)
          if (c & 0x80) {
            all_punct = false;
            break;
          }
        }
        if (all_punct && !core.empty()) return true;

        // 3. Skip Blocklist (Articles, Prepositions, Conjunctions, Linking Verbs)
        // Transform to lowercase for comparison
        std::transform(core.begin(), core.end(), core.begin(), ::tolower);
        static const std::set<std::string> skip_words = {
            // Articles
            "the", "a", "an",
            // Prepositions
            "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as", "into", "like", "through", "after",
            "over", "between", "out", "against", "during", "without", "before", "under", "around", "among",
            // Linking Verbs / Auxiliaries
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "can", "could", "will", "would", "shall",
            "should", "may", "might", "must",
            // Conjunctions
            "and", "but", "or", "nor", "so", "yet", "if", "than", "then", "else", "when", "while", "where", "because", "since",
            "although", "though", "unless"};
        if (skip_words.count(core)) return true;

        return false;
      };

      bool skip_token = should_skip_token(current_token_str);

      // 1. Prefill check
      if (probing_args_.enable_prefill_check && package_cnt == 0) {
        float max_hallu = 0.0f;
        for (auto& [layer_idx, tensor] : probe_ctx.mlp_outputs) {
          if (model_->prefill_probes.count(layer_idx)) {
            float total_prob = 0.0f;
            auto& probes = model_->prefill_probes[layer_idx];
            if (probes.empty()) continue;
            for (auto& probe : probes) total_prob += probe->predict(tensor);
            float halluc_prob = 1.0f - (total_prob / probes.size());

            if (model_->last_probe_results_.size() < 10000) {
              model_->last_probe_results_.push_back({halluc_prob, layer_idx, "prefill", "hallucination", false, -1, -1});
            }

            if (halluc_prob >= probing_args_.prefill_stop_threshold) {
              nlohmann::json stop_resp;
              stop_resp["status"] = "early_exit_hallucination";
              stop_resp["score"] = halluc_prob;
              stop_resp["layer"] = layer_idx;
              stop_resp["phase"] = "prefill";
              callback(stop_resp.dump(), true);
              stop_generating = true;
              throw std::runtime_error("PROBING_INTERRUPT");
            }
          }
        }
        probe_ctx.collecting = false;
      }

      // 2. Decode check (Debounced)
      bool finished = false;
      if (idx == model_->cfg.eos_token_id || package_cnt + 1 >= max_length) finished = true;

      if (package_cnt > 0 && probing_args_.enable_decode_check && !has_confirmed_key_in_decode) {
        bool is_new_potential = false;
        int pos_layer = model_->cfg.num_hidden_layers - 1;
        if (model_->pos_probe_layer_idx != -1) pos_layer = model_->pos_probe_layer_idx;

        if (idx != model_->cfg.eos_token_id && !skip_token && model_->pos_probe && probe_ctx.mlp_outputs.count(pos_layer)) {
          float pos_score = model_->pos_probe->predict(probe_ctx.mlp_outputs[pos_layer]);
          if (pos_score >= probing_args_.pos_threshold) {
            // Found a key token!
            is_new_potential = true;

            // Create/Update candidate
            auto new_cand = std::make_shared<CandidateKey>();
            new_cand->token_idx = (int)package_cnt;
            new_cand->token_id = (int)idx;
            new_cand->score = pos_score;

            // Deep copy activations
            for (auto const& [l, t_src] : probe_ctx.mlp_outputs) {
              Tensor t_dst = Tensor::empty(t_src.shape(), t_src.dtype(), kCPU);
              t_dst.alloc();
              size_t sz = t_src.numel() * (t_src.dtype() == mllm::kFloat32 ? 4 : 2);
              if (t_src.ptr<void>()) memcpy(t_dst.ptr<void>(), t_src.ptr<void>(), sz);
              new_cand->activations[l] = t_dst;
            }

            candidate_key = new_cand;
            debounce_counter = 5;  // Reset window

            std::cout << "[PosCheck] UpdateCandidate: '" << current_token_str << "' (" << pos_score << ")" << std::endl;
          }
        }

        // B. Debounce / Expiration Logic
        bool trigger_hallu_check = false;

        if (!is_new_potential && candidate_key) {
          debounce_counter--;
          if (debounce_counter <= 0) trigger_hallu_check = true;
        }
        if (finished && candidate_key) trigger_hallu_check = true;

        // C. Execution
        if (trigger_hallu_check && candidate_key) {
          std::cout << "[PosCheck] ConfirmKey: Index " << candidate_key->token_idx << std::endl;
          has_confirmed_key_in_decode = true;  // Mark as done for this sentence

          if (model_->last_probe_results_.size() < 10000) {
            Qwen3ProbingForCausalLM::ProbeResult res;
            res.score = candidate_key->score;
            res.layer = pos_layer;
            res.phase = "decode";
            res.type = "pos_check";
            res.is_key_predicted = true;
            res.token_idx = candidate_key->token_idx;
            res.token_id = candidate_key->token_id;
            model_->last_probe_results_.push_back(res);
          }

          // Run Hallu Check on SAVED activations
          for (auto& [layer_idx, tensor] : candidate_key->activations) {
            // Only check Layer 22 for hallucination as mapped to user request
            if (layer_idx != 22) continue;

            if (model_->decode_probes.count(layer_idx)) {
              auto& probes = model_->decode_probes[layer_idx];
              if (probes.empty()) continue;
              float total_prob = 0.0f;
              for (auto& probe : probes) total_prob += probe->predict(tensor);
              float halluc_prob = 1.0f - (total_prob / probes.size());

              if (model_->last_probe_results_.size() < 100000) {
                model_->last_probe_results_.push_back({halluc_prob, layer_idx, "decode", "hallucination", false,
                                                       candidate_key->token_idx, candidate_key->token_id});
              }

              if (halluc_prob >= probing_args_.decode_stop_threshold) {
                nlohmann::json stop_resp;
                stop_resp["status"] = "early_exit_hallucination";
                stop_resp["score"] = halluc_prob;
                stop_resp["layer"] = layer_idx;
                stop_resp["phase"] = "decode";
                callback(stop_resp.dump(), true);
                stop_generating = true;
              }
            }
          }
          candidate_key = nullptr;  // Clear after processing
          if (stop_generating) throw std::runtime_error("PROBING_INTERRUPT");
        }
      }

      //  3. Context reset for next token
      if (!stop_generating) {
        probe_ctx.soft_reset();
        probe_ctx.save_last_token_only = true;
        if (probing_args_.enable_decode_check) {
          probe_ctx.collecting = true;
          int pos_layer = model_->cfg.num_hidden_layers - 1;
          if (model_->pos_probe_layer_idx != -1) pos_layer = model_->pos_probe_layer_idx;

          probe_ctx.target_layers.insert(pos_layer);
          for (auto const& [l, _] : model_->decode_probes) probe_ctx.target_layers.insert(l);
        }
      }

      // Only output if finished AND successful (no hallucination stop)
      if (finished && !stop_generating) { callback(accumulated_output, true); }

      if (!finished) full_seq_idx.push_back(idx);

      package_cnt++;
    };

    try {
      model_->streamGenerate(input, args, wrapped_callback);
    } catch (const std::exception& e) {
      if (std::string(e.what()) != "PROBING_INTERRUPT") std::cerr << e.what() << std::endl;
    }

    auto thinking_end_token_idx = findThinkStartToken(full_seq_idx);
    full_seq_idx.resize(thinking_end_token_idx);
    for (auto& k_vec : k_cache_addrs_) k_vec.resize(thinking_end_token_idx);
    for (auto& v_vec : v_cache_addrs_) v_vec.resize(thinking_end_token_idx);
    cache_->promote(full_seq_idx, k_cache_addrs_, v_cache_addrs_);
    k_cache_addrs_ = {};
    v_cache_addrs_ = {};
    mllm::cpu::idleHpcThreadPool();
  }

  void fromPreTrain(const std::string& model_path) override {
    namespace fs = std::filesystem;
    fs::path root = fs::path(model_path).lexically_normal();
    auto cfg = Qwen3Config((root / "config.json").string());
    model_ = std::make_shared<Qwen3ProbingForCausalLM>(cfg);
    model_->load(mllm::load((root / "model.mllm").string(), ModelFileVersion::kV2));
    tokenizer_ = std::make_shared<Qwen3Tokenizer>((root / "tokenizer.json").string());
    cache_ = std::make_shared<prefix_cache::Cache>(prefix_cache::CacheOptions{
        .radix_tree_options = {.enable_lru_eviction = false,
                               .eviction_threshold = 0.9f,
                               .enable_path_compression = false,
                               .min_compression_length = 2,
                               .transformer_blocks_num = cfg.num_hidden_layers},
        .allocator_options = {.per_k_token_ele = static_cast<size_t>(cfg.head_dim * cfg.num_key_value_heads),
                              .per_v_token_ele = static_cast<size_t>(cfg.head_dim * cfg.num_key_value_heads),
                              .k_dtype = mllm::kFloat32,
                              .v_dtype = mllm::kFloat32,
                              .enable_cuda = false,
                              .cuda_mem_base = 0x100000,
                              .enable_cpu_hierarchy_memory = true,
                              .zen_fs_options = {
                                  .record = false,
                                  .working_dir = ".",
                                  .blob_bits_size = 20,
                                  .page_bits = 7,
                                  .lane_bits = 5,
                                  .per_k_token_ele = static_cast<size_t>(cfg.head_dim * cfg.num_key_value_heads),
                                  .per_v_token_ele = static_cast<size_t>(cfg.head_dim * cfg.num_key_value_heads),
                                  .k_dtype = mllm::kFloat32,
                                  .v_dtype = mllm::kFloat32,
                                  .mmap_type = mllm::prefix_cache::ZenFSBlobMMapType::kAnonymous,
                              }}});
  }

  std::string ltrim(const std::string& s) {
    size_t start = s.find_first_not_of(" \n\r\t\f\v");
    return (start == std::string::npos) ? "" : s.substr(start);
  }
  std::string rtrim(const std::string& s) {
    size_t end = s.find_last_not_of(" \n\r\t\f\v");
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
  }
  std::string trim(const std::string& s) { return rtrim(ltrim(s)); }

  std::string applyChatTemplate(const nlohmann::json& messages, const std::vector<nlohmann::json>& tools = {},
                                bool add_generation_prompt = true, bool enable_thinking = true,
                                const std::string& bos_token = "", const std::string& eos_token = "<|im_end|>") {
    std::ostringstream oss;
    if (!tools.empty()) {
      oss << "<|im_start|>system\n";
      if (!messages.empty() && messages[0].value("role", "") == "system") { oss << messages[0].value("content", "") << "\n\n"; }
      oss << "# Tools\n\nYou may call one or more functions to assist with the user query.\n";
      oss << "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
      for (const auto& tool : tools) { oss << "\n" << tool.dump(); }
      oss << "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
             "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
             "<args-json-object>}\n</tool_call><|im_end|>\n";
    } else {
      if (!messages.empty() && messages[0].value("role", "") == "system") {
        oss << "<|im_start|>system\n" << messages[0].value("content", "") << "<|im_end|>\n";
      }
    }

    size_t last_query_index = messages.empty() ? 0 : messages.size() - 1;
    bool found_last_query = false;
    if (!messages.empty()) {
      for (int i = messages.size() - 1; i >= 0; --i) {
        const auto& msg = messages[i];
        if (msg.value("role", "") == "user" && msg.contains("content") && msg["content"].is_string()) {
          std::string content_str = msg["content"].get<std::string>();
          if (!(content_str.starts_with("<tool_response>")
                && content_str.find("</tool_response>") == content_str.length() - std::string("</tool_response>").length())) {
            last_query_index = i;
            found_last_query = true;
            break;
          }
        }
      }
    }
    if (messages.empty()) { found_last_query = false; }

    for (size_t i = 0; i < messages.size(); ++i) {
      const auto& message = messages[i];
      std::string role = message.value("role", "");
      std::string content;
      if (message.contains("content") && message["content"].is_string()) { content = message["content"].get<std::string>(); }

      if (role == "user" || (role == "system" && i > 0)) {
        oss << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
      } else if (role == "assistant") {
        std::string reasoning_content;
        if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
          reasoning_content = message["reasoning_content"].get<std::string>();
        } else {
          auto think_end_pos = content.find("</think>");
          if (think_end_pos != std::string::npos) {
            auto think_start_pos = content.rfind("<think>", think_end_pos);
            if (think_start_pos != std::string::npos) {
              reasoning_content = content.substr(think_start_pos + 7, think_end_pos - (think_start_pos + 7));
              content = content.substr(think_end_pos + 8);
            }
          }
        }

        oss << "<|im_start|>" << role << "\n";
        if (found_last_query && i > last_query_index) {
          if ((i == messages.size() - 1) || !reasoning_content.empty()) {
            oss << "<think>\n" << trim(reasoning_content) << "\n</think>\n\n" << ltrim(content);
          } else {
            oss << content;
          }
        } else {
          oss << content;
        }

        if (message.contains("tool_calls")) {
          bool is_first_tool = true;
          for (const auto& tool_call_item : message["tool_calls"]) {
            if ((is_first_tool && !content.empty()) || !is_first_tool) { oss << "\n"; }
            is_first_tool = false;
            const nlohmann::json* tool_call_ptr = &tool_call_item;
            if (tool_call_item.contains("function")) { tool_call_ptr = &tool_call_item["function"]; }
            const nlohmann::json& tool_call = *tool_call_ptr;
            oss << "<tool_call>\n{\"name\": \"" << tool_call.value("name", "") << R"(", "arguments": )";
            const auto& args = tool_call["arguments"];
            if (args.is_string()) {
              oss << args.get<std::string>();
            } else {
              oss << args.dump();
            }
            oss << "}\n</tool_call>";
          }
        }
        oss << "<|im_end|>\n";
      } else if (role == "tool") {
        if (i == 0 || messages[i - 1].value("role", "") != "tool") { oss << "<|im_start|>user"; }
        oss << "\n<tool_response>\n" << content << "\n</tool_response>";
        if (i == messages.size() - 1 || messages[i + 1].value("role", "") != "tool") { oss << "<|im_end|>\n"; }
      }
    }

    if (add_generation_prompt) {
      oss << "<|im_start|>assistant\n";
      if (!enable_thinking) { oss << "<think>\n\n</think>\n\n"; }
    }
    return oss.str();
  }

 private:
  std::vector<std::vector<prefix_cache::vp_addr_t>> k_cache_addrs_;
  std::vector<std::vector<prefix_cache::vp_addr_t>> v_cache_addrs_;
  std::shared_ptr<Qwen3ProbingForCausalLM> model_;
  std::shared_ptr<Qwen3Tokenizer> tokenizer_;
  std::shared_ptr<prefix_cache::Cache> cache_;
  ProbingArgs probing_args_;

 public:
  std::vector<Qwen3ProbingForCausalLM::ProbeResult> getLastProbeResults() const {
    if (model_) return model_->last_probe_results_;
    return {};
  }
};

}  // namespace mllm::models::qwen3_probing
