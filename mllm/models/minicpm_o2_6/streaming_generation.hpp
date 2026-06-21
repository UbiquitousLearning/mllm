// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "mllm/mllm.hpp"
#include "mllm/models/minicpm_o2_6/llm_chunk_generation.hpp"
#include "mllm/models/minicpm_o2_6/tokenization_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/tokenization_chattts.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"

// Forward declarations to avoid circular dependencies
namespace mllm::models {
class ChunkGenerator;
struct ChunkGenerationOutput;
}  // namespace mllm::models

namespace mllm::models::minicpmo {
class MiniCPMOForCausalLM;
}

namespace mllm::models::chattts {
class ConditionalChatTTS;
}

namespace mllm::models::dvae {
class DVAE;
}

namespace mllm::models::vocos {
class Vocos;
}

namespace mllm::models::minicpmo {

/**
 * @brief Output structure for streaming generation
 */
struct OmniOutput {
  std::string text;                 // Generated text chunk
  std::optional<Tensor> audio_wav;  // Generated audio waveform
  int32_t sampling_rate = 24000;    // Audio sampling rate
  bool finished = false;            // Whether generation is complete
};

/**
 * @brief Configuration for streaming generation
 */
struct StreamingGenerationConfig {
  int32_t max_new_tokens = 512;
  int32_t min_new_tokens = 0;
  bool sampling = true;
  bool generate_audio = true;
  bool enable_regenerate = false;

  // Sampling parameters
  float top_p = 0.8f;
  int32_t top_k = 100;
  float temperature = 0.7f;
  float repetition_penalty = 1.05f;

  // TTS parameters
  int32_t output_chunk_size = 25;
  bool force_no_stop = false;

  // TTS temperature for different codebooks
  std::vector<float> tts_temperature = {0.1f, 0.3f, 0.1f, 0.3f};
};

/**
 * @brief Iterator for streaming multimodal generation
 *
 * This iterator provides Python-like streaming generation interface:
 * ```cpp
 * auto streamer = model.streaming_generate(config);
 * for (auto& output : streamer) {
 *     // output.text contains text chunk
 *     // output.audio_wav contains audio waveform (if generate_audio=true)
 * }
 * ```
 */
class StreamingGenerator {
 public:
  StreamingGenerator() = delete;
  StreamingGenerator(Tensor& input_ids, const Tensor& position_ids, minicpmo::MiniCPMOForCausalLM& model,
                     minicpmo::MiniCPMOTokenizer& tokenizer, chattts::ChatTTSTokenizer* tts_tokenizer,
                     const StreamingGenerationConfig& config, chattts::ChatTTSConfig& tts_config)
      : model_(model),
        tokenizer_(tokenizer),
        tts_model_(&model.tts_model_),
        vocos_model_(model.vocos_model_),
        tts_tokenizer_(tts_tokenizer),
        config_(config) {
    // Configure chunk generation
    models::ChunkGenerationConfig chunk_config{
        .max_new_tokens = 10, .chunk_size = 5, .do_sample = false, .save_first_chunk_hidden_states = true};

    // Add EOS tokens for MiniCPMO
    auto eos_ids = tokenizer_.convert2Ids({L"<|im_end|>"});
    chunk_config.eos_token_ids.push_back(eos_ids.at<int64_t>({0, 0}));
    eos_ids = tokenizer_.convert2Ids({L"<|endoftext|>"});
    chunk_config.eos_token_ids.push_back(eos_ids.at<int64_t>({0, 0}));

    // Create chunk generator
    chunk_gen_ = ChunkGenerator{&model, &tokenizer, chunk_config};

    auto dtype = kFloat32;
    auto device = kCPU;
    int32_t num_spk_embs = 1;  // config_.num_spk_embs
    condition_length_ = 1 + num_spk_embs + streaming_text_reserved_len_ + 1;
    tts_start_token_len_ = 1 + num_spk_embs;

    tts_past_key_values_ =
        nn::StaticCache(1024, tts_config.num_hidden_layers, tts_config.num_attention_heads, tts_config.num_key_value_heads,
                        tts_config.hidden_size / tts_config.num_attention_heads,  // kv_dims
                        dtype, dtype, device, false);
    // tts token is generated begin at idx of condition_length_ - 1
    tts_past_key_values_.setCurrentSeqCnt(condition_length_ - 1);

    // Initialize audio input ids
    audio_input_ids_ = Tensor::zeros({1, condition_length_, tts_config.num_vq}, kInt64, device);

    streamer_ = chunk_gen_.initialize(input_ids, position_ids).begin();

    for (int i = 0; i < input_ids.shape()[1]; i++) {
      if (input_ids.at<int64_t>({0, i}) == tokenizer.convert2Ids({L"<|spk_bos|>"}).at<int64_t>({0, 0})) { spk_start_idx_ = i; }
      if (input_ids.at<int64_t>({0, i}) == tokenizer.convert2Ids({L"<|spk_eos|>"}).at<int64_t>({0, 0})) { spk_end_idx_ = i; }
    }
  }

  /**
   * @brief Iterator class for streaming generation
   */
  class Iterator {
   public:
    Iterator(StreamingGenerator* generator, bool is_end) : generator_(generator), is_end_(is_end) {
      // Initialize first chunk
      advance();
    }

    OmniOutput& operator*() { return current_output_; }
    const OmniOutput& operator*() const { return current_output_; }
    OmniOutput* operator->() { return &current_output_; }
    const OmniOutput* operator->() const { return &current_output_; }

    Iterator& operator++() {
      advance();
      return *this;
    }

    bool operator!=(const Iterator& other) const { return is_end_ != other.is_end_; }

   private:
    void advance() {
      if (is_end_) return;

      generator_->generate_next(current_output_);
      if (current_output_.finished) { is_end_ = true; }
    }

    StreamingGenerator* generator_;
    bool is_end_;
    OmniOutput current_output_;
  };

  Iterator begin() { return {this, false}; }
  Iterator end() { return {this, true}; }

 private:
  /**
   * @brief Generate next chunk of output
   *
   * Implemented in streaming_generation.cpp
   */
  void generate_next(OmniOutput& output);

  /**
   * @brief Prepare TTS text by tokenization
   *
   * Implemented in streaming_generation.cpp
   */
  std::pair<std::string, int32_t> prepare_tts_text(const std::string& text);

  /**
   * @brief Build streaming attention mask
   *
   * Creates a mask that specifies which text tokens the model can attend to
   * during audio generation. This enables streaming by limiting attention to
   * the currently available text chunks.
   *
   * Implemented in streaming_generation.cpp
   */
  Tensor build_streaming_mask(int32_t tts_token_lens);

  /**
   * @brief Decode mel spectrogram to audio waveform
   *
   * Implemented in streaming_generation.cpp
   */
  Tensor decode_mel_to_audio(const Tensor& mel_spec);

  /**
   * @brief Linear overlap-add for smooth audio concatenation
   *
   * This function applies crossfading between the end of previous audio
   * and the beginning of current audio to avoid clicks and pops.
   *
   * @param prev_wav Previous audio chunk [B, T1]
   * @param curr_wav Current audio chunk [B, T2]
   * @param overlap Number of samples to overlap (default: 2048)
   * @return Pair of (output_chunk, remaining_for_next)
   *
   * Implemented in streaming_generation.cpp
   */
  std::pair<Tensor, Tensor> linear_overlap_add(const Tensor& prev_wav, const Tensor& curr_wav, int32_t overlap = 2048);

 private:
  minicpmo::MiniCPMOForCausalLM& model_;
  minicpmo::MiniCPMOTokenizer& tokenizer_;
  chattts::ConditionalChatTTS* tts_model_ = nullptr;
  chattts::ChatTTSTokenizer* tts_tokenizer_ = nullptr;
  vocos::Vocos* vocos_model_ = nullptr;

  StreamingGenerationConfig config_;

  // Generation state
  Tensor spk_embeds_ = Tensor::nil();
  ChunkGenerator chunk_gen_;
  ChunkGenerator::Iterator streamer_;
  nn::StaticCache tts_past_key_values_;
  Tensor audio_input_ids_;
  Tensor streaming_tts_text_mask_;

  int32_t spk_start_idx_ = 0;
  int32_t spk_end_idx_ = 0;

  int32_t condition_length_ = 0;

  int32_t chunk_idx_ = 0;
  int32_t new_ids_len_ = 0;
  int32_t prev_text_len_ = 0;
  int32_t tts_start_token_len_ = 0;
  int32_t streaming_text_reserved_len_ = 300;
  int32_t streaming_text_chunk_size_ = 10;

  std::string gen_text_;
  std::string gen_text_raw_;
  bool llm_is_finished_ = false;
  bool finished_ = false;

  Tensor prev_wav_ = Tensor::nil();

  std::function<Tensor(const std::string&)> tts_encode_adapter = [this](const std::string& text) {
    std::vector<int64_t> temp = tts_tokenizer_->encode(text);
    return Tensor::fromVector<int64_t>(temp, {1, static_cast<int>(temp.size())}, kInt64);
  };

  std::function<std::string(const Tensor&)> tts_decode_adapter = [this](const Tensor& token_ids) {
    std::vector<int64_t> ids_vec;
    auto token_ids_ptr = token_ids.ptr<int64_t>();
    for (int i = 0; i < token_ids.shape()[1]; ++i) { ids_vec.push_back(token_ids_ptr[i]); }
    return tts_tokenizer_->decode(ids_vec);
  };
};

}  // namespace mllm::models::minicpmo
