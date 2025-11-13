// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/mllm.hpp"
#include "mllm/models/minicpm_o2_6/modeling_chattts.hpp"
#include "mllm/models/vocos/modeling_vocos.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/modeling_siglip.hpp"
#include "mllm/models/minicpm_o2_6/modeling_resampler.hpp"
#include "mllm/models/minicpm_o2_6/modeling_qwen2vl_for_minicpmo.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::minicpmo {

// Audio Projection Layer for projecting audio features to text embedding space
class AudioProjectionLayer : public nn::Module {
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

// TTS Feature Projector
class TTSProjector : public nn::Module {
 public:
  TTSProjector() = default;

  TTSProjector(const std::string& name, int32_t input_dim, int32_t hidden_dim, int32_t output_dim) : nn::Module(name) {
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

// Main MiniCPM-o Model
class MiniCPMOForCausalLM : public models::ARGeneration {
 public:
  explicit MiniCPMOForCausalLM(const MiniCPMOConfig& config)
      : config_(config),
        llm_(createLLMConfig(config)),
        vpm_("vpm", config),
        resampler_("resampler", config.query_num, config.hidden_size, config.num_attention_heads, config.vision_hidden_size) {
    // Initialize KV cache like Qwen2VL
    kv_cache_ = nn::StaticCache(config.max_cache_length, config.num_hidden_layers,
                                config.num_attention_heads,                       // q_heads
                                config.num_key_value_heads,                       // kv_heads
                                config.hidden_size / config.num_attention_heads,  // kv_dims
                                kFloat32,                                         // k_dtype
                                kFloat32,                                         // v_dtype
                                kCPU,                                             // device_type
                                false                                             // use_fa2
    );

    // Set ARGeneration parameters
    eos_token_id_ = config.eos_token_id;
    max_length_ = config.max_cache_length;
  }

  MiniCPMOConfig config_;
  qwen2vl::Qwen2VLForCausalLM llm_;
  SiglipVisionModel vpm_;
  Resampler resampler_;
  // AudioProjectionLayer audio_projection_layer_;
  // TTSProjector tts_projector_;

  // TTS components (optional, loaded separately)
  chattts::ConditionalChatTTS tts_model_;
  vocos::Vocos* vocos_model_ = nullptr;

 private:
  nn::StaticCache kv_cache_;

 private:
  static qwen2vl::Qwen2VLConfig createLLMConfig(const MiniCPMOConfig& config) {
    qwen2vl::Qwen2VLConfig llm_config;
    llm_config.hidden_size = config.hidden_size;
    llm_config.intermediate_size = config.intermediate_size;
    llm_config.num_attention_heads = config.num_attention_heads;
    llm_config.num_key_value_heads = config.num_key_value_heads;
    llm_config.num_hidden_layers = config.num_hidden_layers;
    llm_config.max_position_embeddings = config.max_position_embeddings;
    llm_config.rms_norm_eps = config.rms_norm_eps;
    llm_config.vocab_size = config.vocab_size;
    llm_config.rope_theta = config.rope_theta;
    llm_config.tie_word_embeddings = config.tie_word_embeddings;
    // Set other necessary fields for Qwen2VL compatibility
    return llm_config;
  }

 public:
  void init_tts_module(models::chattts::ChatTTSConfig& chattts_config) {
    tts_model_ = models::chattts::ConditionalChatTTS("tts", chattts_config);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& inputs, const ARGenerationArgs& args) override {
    // In prefill stage, get "input_ids", in decode stage we get "sequence"
    Tensor input_ids;
    if (inputs.count("input_ids")) {
      input_ids = inputs.at("input_ids");
    } else if (inputs.count("sequence")) {
      input_ids = inputs.at("sequence");
    } else {
      MLLM_ERROR("No input_ids or sequence found!");
      return {};
    }

    Tensor pixel_values = inputs.count("pixel_values") ? inputs.at("pixel_values") : Tensor::nil();
    Tensor tgt_sizes = inputs.count("tgt_sizes") ? inputs.at("tgt_sizes") : Tensor::nil();
    Tensor image_bounds = inputs.count("image_bounds") ? inputs.at("image_bounds") : Tensor::nil();
    Tensor audio_features = inputs.count("audio_features") ? inputs.at("audio_features") : Tensor::nil();

    Tensor prev_position_ids = inputs.count("position_ids") ? inputs.at("position_ids") : Tensor::nil();

    auto input_embeddings = llm_.llm.embedding_(input_ids);

    // Process vision inputs if provided - ONLY in prefill stage
    if (!pixel_values.isNil() && !tgt_sizes.isNil() && prev_position_ids.isNil()) {
      auto vision_outputs = vpm_(pixel_values, tgt_sizes)[0];
      auto vision_embeddings = resampler_(vision_outputs, tgt_sizes)[0];
      if (!image_bounds.isNil()) {
        input_embeddings = merge_vision_text_embeddings(input_embeddings, vision_embeddings, image_bounds);
      }
    }

    // Process audio inputs if provided
    // if (!audio_features.isNil()) {
    //     auto audio_embeddings = encode_audio(audio_features);
    //     // TODO: Similarly handle audio embedding insertion
    //     input_embeddings = merge_audio_text_embeddings(input_embeddings, audio_embeddings, sequence);
    // }

    // Create position IDs based on stage
    Tensor position_ids;
    auto seq_len = input_embeddings.shape()[1];

    if (!prev_position_ids.isNil()) {
      // Decode stage: create [3, 1, 1] position_ids for next token
      auto last_pos = *prev_position_ids.offsettedPtr<int64_t>({0, 0, prev_position_ids.shape()[2] - 1});
      // in case chunk prefilling with multiple tokens
      position_ids = Tensor::empty({3, 1, seq_len}, kInt64).alloc();
      for (int d = 0; d < 3; d++) {
        for (int s = 0; s < seq_len; s++) { position_ids.at<int64_t>({d, 0, s}) = last_pos + s + 1; }
      }
    } else {
      auto last_seen_tokens = kv_cache_.getCurrentSeqCnt(0);
      // Prefill stage: create [3, 1, seq_len] position_ids for full sequence
      position_ids = Tensor::empty({3, 1, seq_len}, kInt64).alloc();
      // Simple sequential position IDs for all dimensions
      for (int d = 0; d < 3; d++) {
        for (int s = 0; s < seq_len; s++) { position_ids.at<int64_t>({d, 0, s}) = last_seen_tokens + s; }
      }
    }

    auto head_dim = config_.hidden_size / config_.num_attention_heads;

    auto inv_freq = llm_.llm.getBuffer("inv_freq");

    std::vector<int32_t> empty_mrope_section;

    auto [llm_embedding_sin, llm_embedding_cos] = qwen2vl::makeMultimodalPositionEmbedding(
        position_ids, inv_freq, config_.max_position_embeddings, head_dim, empty_mrope_section);

    auto output = llm_.llm(input_embeddings, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];

    ARGenerationOutputPast result = {{"sequence", output}, {"position_ids", position_ids}};

    if (!pixel_values.isNil()) { result["pixel_values"] = pixel_values; }
    if (!tgt_sizes.isNil()) { result["tgt_sizes"] = tgt_sizes; }
    if (!image_bounds.isNil()) { result["image_bounds"] = image_bounds; }

    return result;
  }

  // Audio encoding: audio_features -> projection -> text embedding space
  // Tensor encode_audio(const Tensor& audio_features) {
  //     // Project audio features to text embedding space
  //     auto projected_audio = audio_projection_layer_(audio_features)[0];
  //     return projected_audio;
  // }

  // TTS feature generation for audio output
  // Tensor generate_tts_features(const Tensor& text_hidden_states) {
  //     // Project text hidden states to TTS feature space
  //     auto tts_features = tts_projector_(text_hidden_states)[0];
  //     return tts_features;
  // }

  Tensor merge_vision_text_embeddings(Tensor& text_embeddings, Tensor& vision_embeddings, Tensor& image_bounds) {
    auto batch_size = text_embeddings.shape()[0];  // text_embeddings: [1, seq_len, embed_dim]
    auto seq_len = text_embeddings.shape()[1];
    auto embed_dim = text_embeddings.shape()[2];
    auto vision_seq_len = vision_embeddings.shape()[1];  // vision_embeddings:[batch_size, query_num, embed_dim]

    if (!image_bounds.isNil() && image_bounds.shape().size() >= 2) {
      auto num_bounds = vision_embeddings.shape()[0];

      for (int b = 0; b < batch_size; ++b) {
        for (int bound_idx = 0; bound_idx < num_bounds; ++bound_idx) {
          int vision_idx = 0;
          auto start_pos = image_bounds.at<int32_t>({bound_idx, 0}) + 1;
          auto end_pos = image_bounds.at<int32_t>({bound_idx, 1}) - 1;
          // exactly replace <unk> tokens between <slice> and </slice>
          for (int pos = start_pos; pos <= end_pos && vision_idx < vision_seq_len; ++pos, ++vision_idx) {
            float* dst_ptr = text_embeddings.offsettedPtr<float>({b, pos, 0});
            const float* src_ptr = vision_embeddings.offsettedPtr<float>({bound_idx, vision_idx, 0});
            std::memcpy(dst_ptr, src_ptr, embed_dim * sizeof(float));
          }
        }
      }
    }
    return text_embeddings;
  }

  Tensor merge_audio_text_embeddings(const Tensor& text_embeddings, const Tensor& audio_embeddings, const Tensor& sequence) {
    // TODO: Similar to vision embedding fusion
    return text_embeddings;
  }
};

}  // namespace mllm::models::minicpmo
