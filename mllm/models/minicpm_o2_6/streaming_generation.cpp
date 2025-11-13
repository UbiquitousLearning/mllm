// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/minicpm_o2_6/streaming_generation.hpp"
#include <cstring>
#include <vector>
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/minicpm_o2_6/modeling_chattts.hpp"
#include "mllm/models/minicpm_o2_6/llm_chunk_generation.hpp"
#include "mllm/models/vocos/modeling_vocos.hpp"

namespace mllm::models::minicpmo {

void StreamingGenerator::generate_next(OmniOutput& output) {
  output.text.clear();
  output.audio_wav = Tensor::nil();
  output.finished = false;

  if (finished_) {
    output.finished = true;
    // yield last wav without smooth
    if (!prev_wav_.isNil()) {
      output.audio_wav = prev_wav_;
      prev_wav_ = Tensor::nil();
    }
    return;
  }

  if (spk_embeds_.isNil()) {
    streamer_ = ++streamer_;

    spk_embeds_ = streamer_.getLastHiddenStates()[make_slice(0), spk_start_idx_ + 1, kAll];

    std::string tts_eos_token = preprocessor::wideString2Utf8String(L"<|tts_eos|>");
    std::string tts_text = streamer_->text;
    size_t pos = streamer_->text.rfind(tts_eos_token);
    if (pos != std::string::npos) { tts_text = streamer_->text.substr(0, pos); }

    gen_text_raw_ += tts_text;
    gen_text_ += tts_text;
  }

  std::string new_text;

  if (!streamer_->finished) {
    ++streamer_;
    new_text = streamer_->text;
    llm_is_finished_ = streamer_->finished;
  }

  size_t tts_eos_pos = new_text.find("<|tts_eos|>");
  if (tts_eos_pos != std::string::npos) {
    new_text = new_text.substr(0, tts_eos_pos);
    llm_is_finished_ = true;
  }

  gen_text_raw_ += new_text;
  gen_text_ += new_text;

  auto [tts_text, tts_token_lens] = prepare_tts_text(gen_text_);

  int32_t current_chunk_start_token;
  if (chunk_idx_ == 0) {
    current_chunk_start_token = 0;
  } else {
    current_chunk_start_token = chunk_idx_ * streaming_text_chunk_size_ + tts_start_token_len_;
  }

  if (config_.generate_audio) {
    int32_t begin, end;
    if (chunk_idx_ == 0) {
      begin = 0;
      end = streaming_text_chunk_size_ + tts_start_token_len_;
    } else {
      begin = chunk_idx_ * streaming_text_chunk_size_ + tts_start_token_len_;
      int32_t condition_length = 1 + 1 + streaming_text_reserved_len_ + 1;
      end = std::min((chunk_idx_ + 1) * streaming_text_chunk_size_ + tts_start_token_len_, condition_length - 1);
    }

    if (llm_is_finished_) { end = tts_token_lens + tts_start_token_len_ + 1; }

    if (end > begin) {
      // Tokenize TTS text
      auto tts_input_ids = tts_encode_adapter(tts_text);

      auto text_input_ids = tts_input_ids[{0, {begin, end}}];  // NOTE: batch size is always 1

      // Create position IDs
      auto batch_size = text_input_ids.shape()[0];
      auto input_len = text_input_ids.shape()[1];
      auto position_ids = Tensor::empty({batch_size, input_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_len; ++i) { position_ids_ptr[b * input_len + i] = begin + i; }
      }

      // Prefill text into TTS model
      int32_t text_start_idx = position_ids.at<mllm_int64_t>({0, 0});
      // NOTE: mllm handling for in-place update of KV cache
      auto sub_kv_cache = nn::SubStaticCache(tts_past_key_values_, 0, text_start_idx);

      tts_model_->prefillText(text_input_ids, position_ids, sub_kv_cache, chunk_idx_ == 0 ? spk_embeds_ : Tensor::nil());
    }

    // Generate audio codes
    Tensor temperature = Tensor::fromVector(config_.tts_temperature, {4}, kFloat32, kCPU);

    // Build streaming mask
    auto streaming_tts_text_mask = build_streaming_mask(tts_token_lens);

    auto outputs = tts_model_->generate(audio_input_ids_, tts_past_key_values_, temperature,
                                        625,  // eos_token
                                        streaming_tts_text_mask, config_.force_no_stop,
                                        10,  // min_new_token
                                        25,  // max_new_token
                                        config_.top_p, config_.top_k);

    // Note: outputs.past_key_values is already updated in-place during generation
    chunk_idx_ += 1;
    // Update state
    audio_input_ids_ = outputs.audio_input_ids;

    // Prepare new ids slice for decoding
    auto new_ids_slice = outputs.new_ids[{kAll, {std::max(new_ids_len_ - 4, 0), kAll}, kAll}].contiguous().squeeze().clone();

    // Decode to mel spectrogram
    std::vector<Tensor> batch_results{new_ids_slice};
    auto mel_spec = tts_model_->decodeToMelSpecs(batch_results).contiguous();

    new_ids_len_ = outputs.new_ids.shape()[1];

    auto wav_np = decode_mel_to_audio(mel_spec);

    // Smooth overlap with previous audio
    if (!prev_wav_.isNil()) {
      auto [current_wav, next_prev] = linear_overlap_add(prev_wav_, wav_np, 2048);
      output.audio_wav = current_wav;
      prev_wav_ = next_prev;
    } else {
      prev_wav_ = wav_np;
    }

    output.text = gen_text_raw_.substr(prev_text_len_);
    prev_text_len_ = gen_text_raw_.length();
    output.sampling_rate = 24000;

    if (outputs.finished || new_ids_len_ > 1024) { finished_ = true; }
  } else if (llm_is_finished_) {
    finished_ = true;
    output.finished = true;
  }
}

Tensor StreamingGenerator::decode_mel_to_audio(const Tensor& mel_spec) {
  if (vocos_model_ == nullptr) {
    MLLM_ERROR("Vocos model not initialized");
    return Tensor::nil();
  }

  // Use Vocos to decode mel to audio
  auto wav = (*vocos_model_).decode({mel_spec}, {})[0];
  return wav;
}

std::pair<std::string, int32_t> StreamingGenerator::prepare_tts_text(const std::string& text) {
  // Tokenize the text without special tokens
  if (tts_tokenizer_ == nullptr) {
    MLLM_ERROR("TTS tokenizer not initialized");
    return {text, static_cast<int32_t>(text.length())};
  }

  // Tokenize text (without special tokens)
  auto tts_tokens = tts_encode_adapter(text);
  int32_t tts_tokens_len = tts_tokens.shape()[1];

  std::string processed_text = text;
  std::string pad_str;

  if (tts_tokens_len < streaming_text_reserved_len_) {
    // Add padding tokens
    int32_t num_pad_tokens = streaming_text_reserved_len_ - tts_tokens_len;
    pad_str = "[Etts]";
    for (int32_t i = 0; i < num_pad_tokens - 1; ++i) { pad_str += "[PAD]"; }
  } else {
    // Truncate to reserved length and decode back
    auto truncated_tokens = tts_tokens[{kAll, {0, streaming_text_reserved_len_}}].contiguous();
    tts_tokens_len = streaming_text_reserved_len_;
    processed_text = tts_decode_adapter(truncated_tokens);
    pad_str = "";
  }

  // Build speaker embedding placeholder
  // num_spk_embs is hardcoded to 1 in the constructor
  int32_t num_spk_embs = 1;
  std::string spk_emb_placeholder_tts;
  for (int32_t i = 0; i < num_spk_embs; ++i) { spk_emb_placeholder_tts += "[spk_emb]"; }

  // Construct new text: [Stts][spk_emb]...<text><pad_str>[Ptts]
  std::string new_text_tts = "[Stts]" + spk_emb_placeholder_tts + processed_text + pad_str + "[Ptts]";

  return {new_text_tts, tts_tokens_len};
}

Tensor StreamingGenerator::build_streaming_mask(int32_t tts_token_lens) {
  // streaming_text_reserved_len = 300
  auto mask = Tensor::zeros({1, streaming_text_reserved_len_}, kFloat32, kCPU);

  // Fill mask: 1 for available tokens, 0 for unavailable
  auto mask_ptr = mask.ptr<float>();
  for (int32_t i = 0; i < std::min(tts_token_lens, streaming_text_reserved_len_); ++i) { mask_ptr[i] = 1.0f; }

  return mask;
}

std::pair<Tensor, Tensor> StreamingGenerator::linear_overlap_add(const Tensor& prev_wav, const Tensor& curr_wav,
                                                                 int32_t overlap) {
  auto batch_size = curr_wav.shape()[0];
  auto curr_len = curr_wav.shape()[1];

  // Ensure overlap doesn't exceed lengths
  overlap = std::min(overlap, static_cast<int32_t>(prev_wav.shape()[1]));
  overlap = std::min(overlap, curr_len);

  // Create output: previous without overlap + crossfaded region
  // This length calculation is correct for your streaming strategy
  auto prev_len = prev_wav.shape()[1];
  auto output_len = prev_len - overlap + overlap;
  auto output = Tensor::zeros({batch_size, output_len}, prev_wav.dtype(), prev_wav.device());

  auto output_ptr = output.ptr<float>();
  auto prev_ptr = prev_wav.ptr<float>();
  auto curr_ptr = curr_wav.ptr<float>();

  // Handle potential divide-by-zero if overlap is 1 or 0
  // (Although with overlap=2048 this is safe, it's good practice)
  const float fade_denominator = (overlap > 1) ? static_cast<float>(overlap - 1) : 1.0f;

  for (int b = 0; b < batch_size; ++b) {
    int out_idx = 0;

    // Copy previous audio without overlap region
    for (int i = 0; i < prev_len - overlap; ++i, ++out_idx) {
      output_ptr[b * output_len + out_idx] = prev_ptr[b * prev_len + i];
    }

    // Crossfade region
    for (int i = 0; i < overlap; ++i, ++out_idx) {
      // **FIXED RAMP:**
      float fade_in = static_cast<float>(i) / fade_denominator;
      float fade_out = 1.0f - fade_in;

      float prev_sample = prev_ptr[b * prev_len + (prev_len - overlap + i)];
      float curr_sample = curr_ptr[b * curr_len + i];

      output_ptr[b * output_len + out_idx] = fade_out * prev_sample + fade_in * curr_sample;
    }
  }

  // Remaining current audio for next iteration
  // This logic is also correct for your streaming strategy
  auto remaining_len = curr_len - overlap;
  auto next_prev = Tensor::zeros({batch_size, remaining_len}, curr_wav.dtype(), curr_wav.device());
  auto next_prev_ptr = next_prev.ptr<float>();

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < remaining_len; ++i) { next_prev_ptr[b * remaining_len + i] = curr_ptr[b * curr_len + (overlap + i)]; }
  }

  return {output, next_prev};
}

}  // namespace mllm::models::minicpmo
