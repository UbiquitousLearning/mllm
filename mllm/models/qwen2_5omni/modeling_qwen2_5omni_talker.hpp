// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "mllm/core/Parallel.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Enumerate.hpp"

#include "mllm/models/qwen2_5omni/configuration_qwen2_5omni.hpp"

namespace mllm::models::qwen2_5omni {

constexpr float kPi = 3.14159265358979323846f;

inline auto makeTalkerRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim); }
  return inv_freq;
}

inline auto makeTalkerPositionEmbedding(Tensor& position_ids, const Tensor& inv_freq, const std::vector<int32_t>& mrope_section)
    -> std::pair<Tensor, Tensor> {
  MLLM_RT_ASSERT_EQ(position_ids.shape().size(), 3);
  MLLM_RT_ASSERT_EQ(position_ids.shape()[1], 1);

  Tensor tmp_sin = Tensor::empty({3, position_ids.shape()[2], inv_freq.shape()[0] * 2}).alloc();
  Tensor tmp_cos = Tensor::empty({3, position_ids.shape()[2], inv_freq.shape()[0] * 2}).alloc();

  for (int b = 0; b < 3; ++b) {
    for (int d = 0; d < inv_freq.shape()[0]; ++d) {
      for (int s = 0; s < position_ids.shape()[2]; ++s) {
        auto value = inv_freq.ptr<float>()[d] * (*position_ids.offsettedPtr<int64_t>({b, 0, s}));
        *tmp_cos.offsettedPtr<float>({b, s, d}) = cosf(value);
        *tmp_cos.offsettedPtr<float>({b, s, d + inv_freq.shape()[0]}) = cosf(value);
        *tmp_sin.offsettedPtr<float>({b, s, d}) = sinf(value);
        *tmp_sin.offsettedPtr<float>({b, s, d + inv_freq.shape()[0]}) = sinf(value);
      }
    }
  }

  Tensor sin = Tensor::nil();
  Tensor cos = Tensor::nil();

  if (!mrope_section.empty()) {
    auto double_rope_section = mrope_section;
    for (int i : mrope_section) { double_rope_section.push_back(i); }

    int num_rows = tmp_sin.shape()[1];
    int num_cols = tmp_sin.shape()[2];

    sin = Tensor::empty({num_rows, num_cols}, kFloat32, kCPU).alloc();
    cos = Tensor::empty({num_rows, num_cols}, kFloat32, kCPU).alloc();

    std::vector<int> start_cols;
    int current_start = 0;
    start_cols.push_back(current_start);
    for (int s : double_rope_section) {
      current_start += s;
      start_cols.push_back(current_start);
    }

    for (int j = 0; j < static_cast<int>(double_rope_section.size()); ++j) {
      int layer = j % 3;
      int s_j = double_rope_section[j];
      int start_col_in = start_cols[j];
      int start_col_out = start_cols[j];
      for (int row = 0; row < num_rows; ++row) {
        auto in_cos_row_ptr = tmp_cos.offsettedPtr<float>({layer, row, 0});
        auto out_cos_row_ptr = cos.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) { out_cos_row_ptr[start_col_out + c] = in_cos_row_ptr[start_col_in + c]; }

        auto in_sin_row_ptr = tmp_sin.offsettedPtr<float>({layer, row, 0});
        auto out_sin_row_ptr = sin.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) { out_sin_row_ptr[start_col_out + c] = in_sin_row_ptr[start_col_in + c]; }
      }
    }
  } else {
    sin = tmp_sin;
    cos = tmp_cos;
  }

  return {sin, cos};
}

struct Qwen2_5OmniSpeakerParams {
  int64_t bos_token = 0;
  Tensor cond = Tensor::nil();
  Tensor ref_mel = Tensor::nil();
};

struct Qwen2_5OmniSpeakerMap {
  std::unordered_map<std::string, Qwen2_5OmniSpeakerParams> speakers;
  std::string default_speaker;
};

inline Tensor tensorFromJson(const nlohmann::ordered_json& obj) {
  if (!obj.contains("shape") || !obj.contains("data")) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Invalid speaker json entry: missing shape/data.");
  }
  auto shape = obj["shape"].get<std::vector<int32_t>>();
  auto data = obj["data"].get<std::vector<float>>();

  int64_t expected = 1;
  for (auto dim : shape) { expected *= dim; }
  MLLM_RT_ASSERT_EQ(expected, static_cast<int64_t>(data.size()));

  Tensor out = Tensor::empty(shape, kFloat32, kCPU).alloc();
  std::copy(data.begin(), data.end(), out.ptr<float>());
  return out;
}

inline Qwen2_5OmniSpeakerMap loadSpeakerMap(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open spk_dict.json at {}", path); }

  nlohmann::ordered_json root;
  in >> root;

  Qwen2_5OmniSpeakerMap map;
  bool first = true;
  for (auto it = root.begin(); it != root.end(); ++it) {
    const auto& name = it.key();
    const auto& entry = it.value();
    Qwen2_5OmniSpeakerParams params;
    params.bos_token = entry.value("bos_token", 0);
    params.cond = tensorFromJson(entry["cond"]);
    params.ref_mel = tensorFromJson(entry["ref_mel"]);
    map.speakers.emplace(name, std::move(params));
    if (first) {
      map.default_speaker = name;
      first = false;
    }
  }

  if (map.speakers.empty()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Empty speaker map in {}", path); }
  return map;
}

class Qwen2_5OmniTalkerMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen2_5OmniTalkerMLP() = default;
  Qwen2_5OmniTalkerMLP(const std::string& name, const Qwen2_5OmniTalkerConfig& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false);
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

class Qwen2_5OmniTalkerAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::MultimodalRoPE q_rope_;
  nn::MultimodalRoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen2_5OmniTalkerAttention() = default;

  Qwen2_5OmniTalkerAttention(const std::string& name, const Qwen2_5OmniTalkerConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    head_dim_ = cfg.head_dim;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, true);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, true);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, true);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, false);

    q_rope_ = reg<nn::MultimodalRoPE>(
        "q_rope", aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                       .max_position_embeddings = cfg.max_position_embeddings,
                                                       .mrope_section = cfg.mrope_section});
    k_rope_ = reg<nn::MultimodalRoPE>(
        "k_rope", aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                       .max_position_embeddings = cfg.max_position_embeddings,
                                                       .mrope_section = cfg.mrope_section});

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    auto [k, v] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = k;
    value_states = v;

    Tensor attn;
    if (key_states.dtype() == kFloat32) {
      attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
    } else if (key_states.dtype() == kFloat16) {
      attn = nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
      attn = attn.to(kFloat16);
    }

    auto output = nn::functional::matmul(attn, value_states);
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);
    return {output};
  }

  int layer_idx_ = 0;
};

class Qwen2_5OmniTalkerDecoder final : public nn::Module {
 public:
  Qwen2_5OmniTalkerAttention self_attn_;
  Qwen2_5OmniTalkerMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen2_5OmniTalkerDecoder() = default;

  Qwen2_5OmniTalkerDecoder(const std::string& name, const Qwen2_5OmniTalkerConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen2_5OmniTalkerAttention>("self_attn", cfg);
    mlp_ = reg<Qwen2_5OmniTalkerMLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    auto x = input_layer_norm_(inputs[0]);
    x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    x = x + tmp;
    return {x};
  }
};

class Qwen2_5OmniTalkerModel final : public nn::Module {
  nn::ModuleList<Qwen2_5OmniTalkerDecoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  Qwen2_5OmniTalkerModel() = default;

  Qwen2_5OmniTalkerModel(const std::string& name, const Qwen2_5OmniTalkerConfig& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<Qwen2_5OmniTalkerDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.embedding_size);

    auto inv = makeTalkerRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0]; }
    x = norm_(x);

    return {x};
  }

  nn::Embedding embedding_;
};

struct Qwen2_5OmniTalkerOutput {
  Tensor logits = Tensor::nil();
  Tensor thinker_reply_part = Tensor::nil();
  Tensor position_ids = Tensor::nil();
};

class Qwen2_5OmniTalker final : public nn::Module {
 public:
  Qwen2_5OmniTalker() = delete;
  Qwen2_5OmniTalker(const std::string& name, const Qwen2_5OmniTalkerConfig& cfg) : nn::Module(name), cfg_(cfg) {
    thinker_to_talker_proj_ = reg<nn::Linear>("thinker_to_talker_proj", cfg.embedding_size, cfg.hidden_size, true);
    model_ = reg<Qwen2_5OmniTalkerModel>("model", cfg);
    codec_head_ = reg<nn::Linear>("codec_head", cfg.hidden_size, cfg.vocab_size, false);

    kv_cache_ = nn::StaticCache(cfg.max_position_embeddings, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads,
                                cfg.head_dim, kFloat32, kFloat32, kCPU, false);

    codec_bos_token_ = cfg.tts_codec_start_token_id;
    codec_eos_token_ = cfg.tts_codec_end_token_id;
    codec_pad_token_ = cfg.tts_codec_pad_token_id;
    codec_mask_token_ = cfg.tts_codec_mask_token_id;
    text_bos_token_ = cfg.tts_text_start_token_id;
    text_eos_token_ = cfg.tts_text_end_token_id;
    text_pad_token_ = cfg.tts_text_pad_token_id;
  }

  void clearCache() {
    kv_cache_.clearCache();
    rope_deltas_ = Tensor::nil();
  }

  Qwen2_5OmniTalkerOutput forward(const Tensor& input_ids, const Tensor& input_text_ids, Tensor thinker_reply_part,
                                  Tensor inputs_embeds, const Tensor& attention_mask, const Tensor& image_grid_thw,
                                  Tensor position_ids) {
    Tensor ids_for_pos = input_text_ids.isNil() ? input_ids : input_text_ids;
    position_ids = getPositionIds(ids_for_pos, image_grid_thw, position_ids);

    const bool prefill = kv_cache_.getCurrentSeqCnt(0) == 0;
    if (!inputs_embeds.isNil() && prefill) {
      const auto S = inputs_embeds.shape()[1];
      MLLM_RT_ASSERT(S >= 2);

      auto bos_token = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
      bos_token.at<mllm_int64_t>({0, 0}) = codec_bos_token_;
      auto bos_embed = model_.embedding_(bos_token);

      auto pad_token = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
      pad_token.at<mllm_int64_t>({0, 0}) = codec_pad_token_;
      auto pad_embed = model_.embedding_(pad_token);

      auto embed_dim = inputs_embeds.shape()[2];
      if (inputs_embeds.dtype() == kFloat32) {
        auto* out_ptr = inputs_embeds.offsettedPtr<float>({0, S - 1, 0});
        auto* pad_ptr = inputs_embeds.offsettedPtr<float>({0, S - 2, 0});
        auto* bos_ptr = bos_embed.ptr<float>();
        auto* pad_src_ptr = pad_embed.ptr<float>();
        for (int d = 0; d < embed_dim; ++d) {
          out_ptr[d] += bos_ptr[d];
          pad_ptr[d] += pad_src_ptr[d];
        }
      } else if (inputs_embeds.dtype() == kFloat16) {
        auto* out_ptr = inputs_embeds.offsettedPtr<mllm_fp16_t>({0, S - 1, 0});
        auto* pad_ptr = inputs_embeds.offsettedPtr<mllm_fp16_t>({0, S - 2, 0});
        auto* bos_ptr = bos_embed.ptr<mllm_fp16_t>();
        auto* pad_src_ptr = pad_embed.ptr<mllm_fp16_t>();
        for (int d = 0; d < embed_dim; ++d) {
          out_ptr[d] = static_cast<mllm_fp16_t>(static_cast<float>(out_ptr[d]) + static_cast<float>(bos_ptr[d]));
          pad_ptr[d] = static_cast<mllm_fp16_t>(static_cast<float>(pad_ptr[d]) + static_cast<float>(pad_src_ptr[d]));
        }
      }
    }

    if (inputs_embeds.isNil()) {
      auto codec_embeds = model_.embedding_(input_ids);
      inputs_embeds = codec_embeds + thinker_reply_part[{kAll, {0, 1}, kAll}];
      if (thinker_reply_part.shape()[1] > 1) {
        thinker_reply_part = thinker_reply_part[{kAll, {1, thinker_reply_part.shape()[1]}, kAll}];
      }
    }

    auto [llm_embedding_sin, llm_embedding_cos] =
        makeTalkerPositionEmbedding(position_ids, model_.getBuffer("inv_freq"), cfg_.mrope_section);

    auto talker_lm_input = thinker_to_talker_proj_(inputs_embeds);
    auto hidden_states = model_(talker_lm_input, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];
    auto logits = codec_head_(hidden_states).to(kFloat32);

    return {
        .logits = logits,
        .thinker_reply_part = thinker_reply_part,
        .position_ids = position_ids,
    };
  }

  int64_t codec_bos_token() const { return codec_bos_token_; }
  int64_t codec_eos_token() const { return codec_eos_token_; }
  int64_t codec_pad_token() const { return codec_pad_token_; }
  int64_t codec_mask_token() const { return codec_mask_token_; }
  int64_t text_eos_token() const { return text_eos_token_; }
  int64_t text_pad_token() const { return text_pad_token_; }
  int64_t text_bos_token() const { return text_bos_token_; }

  Qwen2_5OmniTalkerModel model_;

 private:
  Tensor getPositionIds(const Tensor& input_ids, const Tensor& image_grid_thw, const Tensor& position_ids) const {
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);

    bool has_multimodal = false;
    auto input_ids_ptr = input_ids.ptr<int64_t>();
    auto seq_len = input_ids.shape()[1];
    for (int s = 0; s < seq_len; ++s) {
      if (input_ids_ptr[s] == cfg_.vision_start_token_id || input_ids_ptr[s] == cfg_.audio_start_token_id) {
        has_multimodal = true;
        break;
      }
    }

    if (has_multimodal) { return getPositionIdsPrefill(input_ids, image_grid_thw); }

    if (!position_ids.isNil()) {
      auto last_pos = position_ids.constAt<int64_t>({0, 0, position_ids.shape()[2] - 1});
      auto ret_position_ids = Tensor::empty({3, 1, 1}, kInt64, kCPU).alloc();
      *ret_position_ids.offsettedPtr<int64_t>({0, 0, 0}) = last_pos + 1;
      *ret_position_ids.offsettedPtr<int64_t>({1, 0, 0}) = last_pos + 1;
      *ret_position_ids.offsettedPtr<int64_t>({2, 0, 0}) = last_pos + 1;
      return ret_position_ids;
    }

    auto B = input_ids.shape()[0];
    auto S = seq_len;
    MLLM_RT_ASSERT_EQ(B, 1);

    Tensor out = Tensor::empty({3, B, S}, kInt64, kCPU).alloc();
    for (int d = 0; d < 3; ++d) {
      auto out_ptr = out.offsettedPtr<int64_t>({d, 0, 0});
      for (int64_t s = 0; s < S; ++s) { out_ptr[s] = s; }
    }
    return out;
  }

  Tensor getPositionIdsPrefill(const Tensor& input_ids, const Tensor& image_grid_thw) const {
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);

    auto B = input_ids.shape()[0];
    auto S = input_ids.shape()[1];
    MLLM_RT_ASSERT_EQ(B, 1);

    Tensor position_ids = Tensor::empty({3, B, S}, kInt64, kCPU).alloc();
    auto input_ids_ptr = input_ids.ptr<int64_t>();

    auto fill_text_positions = [&](int start_seq, int len, int64_t start_id) {
      for (int d = 0; d < 3; ++d) {
        auto out_ptr = position_ids.offsettedPtr<int64_t>({d, 0, 0});
        for (int i = 0; i < len; ++i) { out_ptr[start_seq + i] = start_id + i; }
      }
    };

    int seq_idx = 0;
    int image_idx = 0;
    int64_t current_max_position_id = -1;
    const int total_images = image_grid_thw.isNil() ? 0 : image_grid_thw.shape()[0];

    while (seq_idx < S) {
      int next_vision = -1;
      int next_audio = -1;
      for (int i = seq_idx; i < S; ++i) {
        if (input_ids_ptr[i] == cfg_.vision_start_token_id) {
          next_vision = i;
          break;
        }
      }
      for (int i = seq_idx; i < S; ++i) {
        if (input_ids_ptr[i] == cfg_.audio_start_token_id) {
          next_audio = i;
          break;
        }
      }

      if (next_vision == -1 && next_audio == -1) {
        const int text_len = S - seq_idx;
        if (text_len > 0) { fill_text_positions(seq_idx, text_len, current_max_position_id + 1); }
        break;
      }

      const bool is_vision = (next_vision != -1) && (next_audio == -1 || next_vision < next_audio);
      const int segment_start = is_vision ? next_vision : next_audio;

      const int text_len = segment_start - seq_idx;
      if (text_len > 0) {
        fill_text_positions(seq_idx, text_len, current_max_position_id + 1);
        current_max_position_id += text_len;
      }

      if (is_vision) {
        fill_text_positions(segment_start, 1, current_max_position_id + 1);
        current_max_position_id += 1;

        int vision_end = -1;
        for (int i = segment_start + 1; i < S; ++i) {
          if (input_ids_ptr[i] == cfg_.vision_end_token_id) {
            vision_end = i;
            break;
          }
        }
        MLLM_RT_ASSERT(vision_end != -1);

        if (image_idx >= total_images) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Image index out of range."); }

        auto grid_t = image_grid_thw.ptr<int32_t>()[image_idx * 3];
        auto grid_h = image_grid_thw.ptr<int32_t>()[image_idx * 3 + 1];
        auto grid_w = image_grid_thw.ptr<int32_t>()[image_idx * 3 + 2];
        int vision_len = grid_t * grid_h * grid_w;
        vision_len /= (cfg_.spatial_merge_size * cfg_.spatial_merge_size);

        for (int i = 0; i < vision_len; ++i) {
          const int pos = segment_start + 1 + i;
          if (pos >= S) { break; }
          for (int d = 0; d < 3; ++d) {
            *position_ids.offsettedPtr<int64_t>({d, 0, pos}) = current_max_position_id + 1 + i;
          }
        }
        current_max_position_id += vision_len;

        fill_text_positions(vision_end, 1, current_max_position_id + 1);
        current_max_position_id += 1;

        seq_idx = vision_end + 1;
        image_idx += 1;
      } else {
        fill_text_positions(segment_start, 1, current_max_position_id + 1);
        current_max_position_id += 1;

        int audio_end = -1;
        for (int i = segment_start + 1; i < S; ++i) {
          if (input_ids_ptr[i] == cfg_.audio_end_token_id) {
            audio_end = i;
            break;
          }
        }
        MLLM_RT_ASSERT(audio_end != -1);

        std::vector<int32_t> audio_positions;
        for (int i = segment_start + 1; i < audio_end; ++i) {
          if (input_ids_ptr[i] == cfg_.audio_token_id) {
            audio_positions.push_back(i);
          } else {
            MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported token inside audio segment.");
          }
        }
        const int audio_len = static_cast<int>(audio_positions.size());
        if (audio_len == 0) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Empty audio tokens inside audio segment."); }
        const int64_t audio_start_id = current_max_position_id + 1;
        for (int i = 0; i < audio_len; ++i) {
          const int64_t pos_id = audio_start_id + i;
          for (int d = 0; d < 3; ++d) {
            *position_ids.offsettedPtr<int64_t>({d, 0, audio_positions[i]}) = pos_id;
          }
        }
        current_max_position_id += audio_len;
        fill_text_positions(audio_end, 1, current_max_position_id + 1);
        current_max_position_id += 1;
        seq_idx = audio_end + 1;
      }
    }

    return position_ids;
  }

  const Qwen2_5OmniTalkerConfig& cfg_;
  nn::Linear thinker_to_talker_proj_;
  nn::Linear codec_head_;
  nn::StaticCache kv_cache_;
  Tensor rope_deltas_ = Tensor::nil();

  int64_t codec_bos_token_ = 0;
  int64_t codec_eos_token_ = 0;
  int64_t codec_pad_token_ = 0;
  int64_t codec_mask_token_ = 0;
  int64_t text_bos_token_ = 0;
  int64_t text_eos_token_ = 0;
  int64_t text_pad_token_ = 0;
};

}  // namespace mllm::models::qwen2_5omni
