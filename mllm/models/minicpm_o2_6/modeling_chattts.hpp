// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/llama/configuration_llama.hpp"
#include "mllm/models/llama/modeling_llama.hpp"
#include "mllm/models/minicpm_o2_6/configuration_chattts.hpp"
#include "mllm/models/minicpm_o2_6/modeling_dvae.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/layers/Embedding.hpp"
#include "mllm/nn/layers/ReLU.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstdlib>
#include <ctime>

namespace mllm::models::chattts {

inline auto makeLLaMARotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                        float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;

  // Create freqs tensor: position_ids @ inv_freq
  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  // Compute freqs = position_ids[:, :, None] @ inv_freq[None, :]
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }

  // Create sin and cos tensors with shape [batch_size, seq_len, dim]
  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  // Compute sin and cos embeddings: emb = [freqs, freqs]
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;

        // Store the same values in both halves: [freqs, freqs]
        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

/**
 * MultiModal Projector for LLM hidden states to TTS embedding space
 */
class MultiModalProjector final : public nn::Module {
  nn::Linear linear1_;
  nn::Linear linear2_;
  nn::ReLU activation_;

 public:
  MultiModalProjector() = default;

  MultiModalProjector(const std::string& name, int32_t input_dim, int32_t output_dim) : nn::Module(name) {
    // MLP projector: input_dim -> output_dim*4 -> output_dim
    linear1_ = reg<nn::Linear>("linear1", input_dim, output_dim, true);
    activation_ = reg<nn::ReLU>("relu");
    linear2_ = reg<nn::Linear>("linear2", output_dim, output_dim, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = linear1_(inputs[0]);
    x = activation_(x);
    x = linear2_(x);
    return {x};
  }
};

/**
 * ConditionalChatTTS: A conditional text-to-speech model that can generate speech from text with speaker conditioning.
 *
 * This model extends ARGeneration to provide text-to-speech capabilities with:
 * - LLM hidden state conditioning
 * - Streaming generation
 * - Multiple VQ codebook support
 *
 */
class ConditionalChatTTS : public nn::Module {
  ChatTTSConfig cfg_;

  // Core TTS components
  MultiModalProjector projector_;        // in minicpm-o2-6, projector is always mlp
  std::vector<nn::Embedding> emb_code_;  // Multiple VQ codebooks
  nn::Embedding emb_text_;
  std::vector<nn::Linear> head_code_;  // Multiple VQ heads

  dvae::DVAE dvae_;

  llama::LlamaText model_;

 public:
  explicit ConditionalChatTTS(const std::string& name, const ChatTTSConfig& cfg) : cfg_(cfg), nn::Module(name) {
    MLLM_RT_ASSERT(cfg.use_mlp);  // in minicpm, MLP is always used
    projector_ = reg<MultiModalProjector>("projector", cfg.llm_dim, cfg.hidden_size);

    for (auto i = 0; i < cfg.num_vq; ++i) {
      emb_code_.emplace_back(reg<nn::Embedding>("emb_code." + std::to_string(i), cfg.num_audio_tokens, cfg.hidden_size));
    }

    emb_text_ = reg<nn::Embedding>("emb_text", cfg.num_text_tokens, cfg.hidden_size);

    for (auto i = 0; i < cfg.num_vq; ++i) {
      head_code_.emplace_back(reg<nn::Linear>("head_code." + std::to_string(i) + ".parametrizations", cfg.hidden_size,
                                              cfg.num_audio_tokens, false));
    }

    dvae_ = reg<dvae::DVAE>("dvae");

    auto llama_cfg = mllm::models::llama::LLaMAConfig();
    llama_cfg.hidden_size = cfg.hidden_size;
    llama_cfg.intermediate_size = cfg.intermediate_size;
    llama_cfg.num_attention_heads = cfg.num_attention_heads;
    llama_cfg.num_key_value_heads = cfg.num_key_value_heads;
    llama_cfg.num_hidden_layers = cfg.num_hidden_layers;
    llama_cfg.max_position_embeddings = cfg.max_position_embeddings;
    model_ = reg<llama::LlamaText>("model", llama_cfg);
  }

  /**
   * Merge input_ids and speaker embeddings into input embeddings
   */
  Tensor mergeInputsEmbeds(const Tensor& input_ids, const Tensor& lm_spk_emb_last_hidden_states) {
    // Embed input_ids to input_embeds
    auto inputs_embeds = emb_text_(input_ids);

    MLLM_RT_ASSERT(cfg_.use_speaker_embedding);

    // Project spk emb to tts hidden size first, [batch_size, num_spk_emb, llm_dim]->[batch_size, num_spk_emb,
    // self.hidden_size]
    auto projected_spk_emb = projector_(lm_spk_emb_last_hidden_states)[0];

    // normalize:  F.normalize(projected_spk_emb, p=2, dim=-1)
    {
      auto ptr = projected_spk_emb.ptr<float>();
      auto S = projected_spk_emb.shape()[0];
      auto D = projected_spk_emb.shape()[1];
      for (int s = 0; s < S; ++s) {
        float norm = 0.0f;
        for (int d = 0; d < D; ++d) {
          auto val = ptr[s * D + d];
          norm += val * val;
        }
        norm = std::sqrt(norm) + 1e-12f;
        for (int d = 0; d < D; ++d) { ptr[s * D + d] /= norm; }
      }
    }

    applySpeakerEmbedding(input_ids, projected_spk_emb, inputs_embeds, cfg_.spk_emb_token_id, cfg_.num_spk_embs);

    return inputs_embeds;
  }

  /**
   * Prefill a chunk of new text tokens in streaming setting
   */
  nn::AbstractStaticCache prefillText(const Tensor& input_ids, Tensor& position_ids, nn::AbstractStaticCache& past_key_values,
                                      const Tensor& lm_spk_emb_last_hidden_states) {
    // 仅支持 batch_size == 1
    MLLM_RT_ASSERT(input_ids.shape()[0] == 1);

    // Merge text and LLM embeddings
    auto inputs_embeds = mergeInputsEmbeds(input_ids, lm_spk_emb_last_hidden_states);

    // because for text, it is standard causal attention mask, do nothing
    Tensor causal_mask = Tensor::nil();

    // Generate RoPE embeddings using the inv_freq buffer
    auto [llm_embedding_sin, llm_embedding_cos] =
        llama::makeRotaryPosEmbedding(position_ids, model_.getBuffer("inv_freq"), 1.0f);
    auto outputs = model_(inputs_embeds, llm_embedding_sin, llm_embedding_cos, causal_mask, AnyValue(&past_key_values));

    return past_key_values;
  }

  /**
   * Prefill audio tokens for sliding window generation
   */
  nn::AbstractStaticCache prefillAudioIds(Tensor& input_ids, nn::AbstractStaticCache& past_key_values,
                                          const Tensor& streaming_tts_text_mask, bool add_audio_bos = true) {
    NYI("not yet verified, if this occurs, check correctness");
    MLLM_RT_ASSERT(input_ids.shape()[0] == 1);

    // Combine VQ codebook embeddings
    std::vector<Tensor> code_embs;
    for (int i = 0; i < cfg_.num_vq; ++i) {
      auto vq_codes = input_ids[{-1, -1, i}].contiguous();

      code_embs.push_back(emb_code_[i](vq_codes));
    }

    // Sum embeddings from all VQ layers
    auto inputs_embeds = code_embs[0];
    for (int i = 1; i < cfg_.num_vq; ++i) { inputs_embeds = inputs_embeds + code_embs[i]; }
    auto input_len = input_ids.shape()[1];

    if (add_audio_bos) {
      auto bos_token = Tensor::zeros({1, 1}, input_ids.dtype(), input_ids.device());
      bos_token.at<int32_t>({0, 0}) = cfg_.audio_bos_token_id;

      auto bos_embeds = emb_text_(bos_token);
      inputs_embeds = nn::functional::concat({bos_embeds, inputs_embeds}, 1);
      input_len += 1;
    }

    // Generate position_ids based on past_key_values length
    auto past_key_values_length = past_key_values.getCurrentSeqCnt(0);
    auto batch_size = inputs_embeds.shape()[0];
    auto position_ids = Tensor::empty({batch_size, input_len}, kInt64, kCPU).alloc();
    auto position_ids_ptr = position_ids.ptr<int64_t>();
    // Fill position_ids: arange(past_key_values_length, past_key_values_length + input_len)
    for (int b = 0; b < batch_size; ++b) {
      for (int i = 0; i < input_len; ++i) { position_ids_ptr[b * input_len + i] = past_key_values_length + i; }
    }

    auto [llm_embedding_sin, llm_embedding_cos] = makeLLaMARotaryPosEmbedding(position_ids, model_.getBuffer("inv_freq"), 1.0f);

    Tensor causal_mask = Tensor::nil();
    causal_mask = makeStreamingChunkMask(inputs_embeds, past_key_values.getCurrentSeqCnt(0), streaming_tts_text_mask,
                                         cfg_.streaming_text_reserved_len, cfg_.streaming_text_chunk_size);

    auto outputs = model_(inputs_embeds, llm_embedding_sin, llm_embedding_cos, causal_mask, AnyValue(&past_key_values));

    return past_key_values;
  }

  /**
   * Generation output structure
   */
  struct GenerationOutput {
    Tensor new_ids;                           // Generated audio codes
    Tensor audio_input_ids;                   // Full audio sequence for update
    nn::AbstractStaticCache past_key_values;  // Updated KV cache
    bool finished;                            // Whether generation is complete
  };

  /**
   * Generate audio codes in streaming or non-streaming setting
   */
  GenerationOutput generate(Tensor& input_ids, nn::AbstractStaticCache& past_key_values, const Tensor& temperature,
                            int32_t eos_token, const Tensor& streaming_tts_text_mask, bool force_no_stop = false,
                            int32_t min_new_token = 10, int32_t max_new_token = 50, float top_p = 0.7f, int top_k = 20,
                            bool do_sample = true) {
    // 仅支持 batch_size == 1
    MLLM_RT_ASSERT(input_ids.shape()[0] == 1);

    // Pre-allocate output buffer
    auto batch_size = input_ids.shape()[0];
    auto seq_len = input_ids.shape()[1];
    auto num_vq = input_ids.shape()[2];

    // Create output buffer: [batch_size, seq_len + max_new_token, num_vq] for pre allocate
    // TODO: this could be optimized in mllm
    auto input_ids_buf = Tensor::zeros({batch_size, seq_len + max_new_token, num_vq}, input_ids.dtype(), input_ids.device());

    bool finished = false;
    int progress = input_ids.shape()[1];
    int condition_length = 1 + (cfg_.use_speaker_embedding ? cfg_.num_spk_embs : 0) + cfg_.streaming_text_reserved_len + 1;
    int start_idx = 1 + (cfg_.use_speaker_embedding ? cfg_.num_spk_embs : 0) + cfg_.streaming_text_reserved_len + 1;

    // Copy existing input_ids to buffer using slice and copy2
    // Equivalent to PyTorch: input_ids_buf.narrow(1, 0, seq_len).copy_(input_ids)
    input_ids_buf[{kAll, {0, progress}, kAll}].copy2(input_ids);
    input_ids.delete_();
    input_ids = input_ids_buf[{kAll, {0, progress}, kAll}];

    for (int i = 0; i < max_new_token && !finished; ++i) {
      // if the first audio token
      bool audio_bos = (progress == condition_length);

      Tensor inputs_embeds;
      if (audio_bos) {
        // Generate the first token, activate the model with `self.audio_bos_token_id`, the model will predict a new audio
        // token. This is a special case because without the `audio bos token`, it is impossible to generate the first audio
        // token in our streaming setting.
        auto bos_token = Tensor::zeros({1, 1}, kInt64);
        bos_token.at<int64_t>({0, 0}) = cfg_.audio_bos_token_id;
        inputs_embeds = emb_text_(bos_token);
      } else {
        // Generate the following audio tokens, it is applicable to all other cases,
        // including second and the following calling of generate.
        // Get the embedding of the 'last' token from input_ids_buf
        std::vector<Tensor> code_embs;
        for (int vq_i = 0; vq_i < cfg_.num_vq; ++vq_i) {
          // get the last token for each vq codebook
          // TODO: maybe it can be not contiguous
          auto last_vq_codes = input_ids_buf[{kAll, {progress - 1, progress}, {vq_i, vq_i + 1}}].contiguous();
          last_vq_codes = last_vq_codes.view({1, 1});  // reshape to [1, 1]
          code_embs.push_back(emb_code_[vq_i](last_vq_codes));
        }
        if (code_embs.empty()) break;

        inputs_embeds = code_embs[0];
        for (int vq_i = 1; vq_i < cfg_.num_vq; ++vq_i) { inputs_embeds = inputs_embeds + code_embs[vq_i]; }
      }

      // Generate position_ids for prefill phase
      auto position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) { position_ids_ptr[b] = past_key_values.getCurrentSeqCnt(0); }

      auto [llm_embedding_sin, llm_embedding_cos] =
          makeLLaMARotaryPosEmbedding(position_ids, model_.getBuffer("inv_freq"), 1.0f);

      Tensor causal_mask = Tensor::nil();
      causal_mask = makeStreamingChunkMask(inputs_embeds, past_key_values.getCurrentSeqCnt(0), streaming_tts_text_mask,
                                           cfg_.streaming_text_reserved_len, cfg_.streaming_text_chunk_size);

      auto outputs = model_(inputs_embeds, llm_embedding_sin, llm_embedding_cos, causal_mask, AnyValue(&past_key_values));
      auto hidden_states = outputs[0];

      // Project hidden states to VQ code logits
      std::vector<Tensor> vq_logits;  // [1, 1, codebook_size(626)]
      vq_logits.reserve(cfg_.num_vq);
      for (int vq_i = 0; vq_i < cfg_.num_vq; ++vq_i) { vq_logits.push_back(head_code_[vq_i](hidden_states)); }

      auto logits = nn::functional::concat(vq_logits, 1);  // [1, num_vq, codebook_size]
      logits = logits.view({-1, logits.shape()[2]});       // [num_vq, codebook_size]

      logits = logits / temperature;

      auto input_ids_sliced = input_ids[{kAll, {start_idx, kAll}, kAll}].permute({0, 2, 1}).contiguous();

      auto logits_token = input_ids_sliced.view({
          input_ids_sliced.shape()[0] * input_ids_sliced.shape()[1],
          -1,
      });

      // Use the parameters passed to the function
      // Apply min_new_token constraint - mask EOS token if we haven't generated enough tokens
      if ((progress - condition_length) < min_new_token) {
        // Mask EOS token by setting its logits to -inf
        auto logits_ptr = logits.ptr<float>();
        auto vocab_size = logits.shape()[1];
        auto num_vq = logits.shape()[0];

        for (int vq_i = 0; vq_i < num_vq; ++vq_i) {
          logits_ptr[vq_i * vocab_size + eos_token] = -std::numeric_limits<float>::infinity();
        }
      }

      // Apply force_no_stop constraint
      if (force_no_stop) {
        auto logits_ptr = logits.ptr<float>();
        auto vocab_size = logits.shape()[1];
        auto num_vq = logits.shape()[0];

        for (int vq_i = 0; vq_i < num_vq; ++vq_i) {
          logits_ptr[vq_i * vocab_size + eos_token] = -std::numeric_limits<float>::infinity();
        }
      }

      // Apply softmax to get probabilities: [num_vq, codebook_size]
      auto scores = nn::functional::softmax(logits.view({1, 1, logits.shape()[0], logits.shape()[1]}), -1).squeeze();
      logits.delete_();  // Free memory

      // Sample from each VQ codebook inependently
      // TODO: here may not consistent with pytorch, need to check correctness in the future
      auto next_tokens = input_ids_buf[{kAll, {progress, progress + 1}, kAll}];
      for (int vq_i = 0; vq_i < cfg_.num_vq; ++vq_i) {
        // Extract logits for current VQ codebook: [codebook_size]
        auto vq_scores = scores[{vq_i, kAll}].contiguous().squeeze();

        int64_t sampled_token;
        if (do_sample) {
          if (top_k > 0) {
            // Use top-k sampling
            sampled_token = sampleTopKSingle(vq_scores, top_k);
          } else if (top_p > 0.0f) {
            // Use top-p sampling
            sampled_token = sampleTopPSingle(vq_scores, top_p);
          } else {
            // Use multinomial sampling
            sampled_token = multinomialSample(vq_scores);
          }
        } else {
          // Greedy sampling
          sampled_token = greedySample(vq_scores);
        }

        next_tokens.at<int64_t>({0, 0, vq_i}) = sampled_token;
        if (sampled_token == eos_token) { finished = true; }
      }

      scores.delete_();  // Free memory

      progress++;
      input_ids = input_ids_buf[{kAll, {0, progress}, kAll}];
    }

    // Extract generated tokens
    Tensor generated_input_ids;
    if (finished) {
      // Remove the last EOS token: [condition_length : progress-1]
      generated_input_ids = input_ids_buf[{kAll, {condition_length, progress - 1}, kAll}];
    } else {
      // Include all generated tokens: [condition_length : progress]
      generated_input_ids = input_ids_buf[{kAll, {condition_length, progress}, kAll}];
    }

    return GenerationOutput{.new_ids = generated_input_ids,
                            .audio_input_ids = input_ids_buf[{kAll, {0, progress}, kAll}],
                            .past_key_values = past_key_values,
                            .finished = finished};
  }

  /**
   * Decode discrete audio codes to mel spectrograms
   *
   * Borrowed from ChatTTS implementation. Converts discrete audio codes from generate()
   * into mel spectrograms using the DVAE decoder.
   *
   * @param result_list List of audio code tensors from generate(), each with shape [seq_len, num_vq]
   * @return Mel spectrograms tensor
   */
  Tensor decodeToMelSpecs(const std::vector<Tensor>& result_list) {
    if (result_list.empty()) { return Tensor::empty({0}, kFloat32); }

    // Find maximum sequence length across all results
    int max_len = 0;
    for (const auto& result : result_list) { max_len = std::max(max_len, static_cast<int>(result.shape()[0])); }

    // Create batch tensor: [batch_size, num_vq, max_len]
    // Note: PyTorch version uses (len(result_list), result_list[0].size(1), max_x_len)
    // which corresponds to [batch_size, num_vq, max_len]
    auto batch_size = static_cast<int>(result_list.size());
    auto num_vq = static_cast<int>(result_list[0].shape()[1]);  // num_vq is the second dimension
    auto batch_result = Tensor::zeros({batch_size, num_vq, max_len}, result_list[0].dtype(), result_list[0].device());

    // Copy and transpose each result into the batch tensor
    for (const auto& [i, src] : enumerate(result_list)) {
      auto src_len = static_cast<int>(src.shape()[0]);
      auto src_num_vq = static_cast<int>(src.shape()[1]);

      MLLM_RT_ASSERT(src_num_vq == num_vq);  // Ensure consistent num_vq across all results

      // PyTorch: batch_result[i].narrow(1, 0, src_len).copy_(src.permute(1, 0))
      // This means: transpose src from [seq_len, num_vq] to [num_vq, seq_len]
      // and copy to batch_result[i][:, :src_len]

      // Manually transpose and copy: src[s, v] -> batch_result[i, v, s]
      auto src_ptr = src.ptr<int32_t>();  // Assuming int32 for audio codes
      auto batch_ptr = batch_result.ptr<int32_t>();

      for (int v = 0; v < num_vq; ++v) {
        for (int s = 0; s < src_len; ++s) {
          // src: [seq_len, num_vq] -> src[s * num_vq + v]
          // batch_result: [batch_size, num_vq, max_len] -> batch_result[i * num_vq * max_len + v * max_len + s]
          auto src_idx = s * num_vq + v;
          auto batch_idx = i * num_vq * max_len + v * max_len + s;
          batch_ptr[batch_idx] = src_ptr[src_idx];
        }
      }
    }

    // Decode through DVAE
    // DVAE expects input and returns mel spectrograms
    auto mel_specs = dvae_(batch_result)[0];
    return mel_specs;
  }

 private:
  /**
   * Apply speaker embeddings to input embeddings at specified positions
   */
  void applySpeakerEmbedding(const Tensor& input_ids, const Tensor& spk_emb, Tensor& input_embeds, int32_t spk_emb_token_id,
                             int32_t num_spk_embs) {
    // Get batch size from input_ids (shape [batch_size, seq_len_max])
    int batch_size = input_ids.shape()[0];

    for (int idx = 0; idx < batch_size; idx++) {
      // Extract tensors for current batch item
      // input_ids_ has shape [seq_len_max]
      auto input_ids_ = input_ids[{idx, kAll}];

      // Find positions where input_ids_ equals spk_emb_token_id
      int32_t spk_emb_token_start = -1;
      {
        auto S = input_ids_.shape()[1];
        auto input_ids_ptr = input_ids_.ptr<int64_t>();
        for (int s = 0; s < S; ++s) {
          if (input_ids_ptr[s] == spk_emb_token_id) {
            spk_emb_token_start = s;
            break;
          }
        }
        MLLM_RT_ASSERT(spk_emb_token_start != -1);
      }

      // Replace the embeddings in input_embeds at positions [spk_emb_token_start : spk_emb_token_start + num_spk_embs]
      // with the speaker embeddings from spk_emb_
      // spk_emb_ has shape [num_spk_emb, hidden_dim]
      auto spk_emb_ = spk_emb[{idx, kAll}];
      auto hidden_dim = spk_emb_.shape()[1];

      // Copy speaker embeddings to input embeddings
      spk_emb_.copy2(input_embeds[{idx, {spk_emb_token_start, spk_emb_token_start + num_spk_embs, 1}, kAll}]);
    }
  }

  /**
   * Create streaming chunk mask for generation
   *
   * In streaming audio generation, this function creates a mask that allows the model
   * to attend to a specific chunk of text tokens when generating each chunk of audio tokens.
   *
   * @param inputs_embeds Input embeddings tensor
   * @param past_seen_tokens Number of tokens already seen by the model
   * @param streaming_tts_text_mask Mask for the text tokens
   * @param streaming_reserved_length Number of reserved tokens for streaming (default 300)
   * @param streaming_text_chunk_size Size of each text chunk (default 10)
   * @return Causal mask for streaming TTS generation, shape [1, 1, seq_len, past_seen_tokens + seq_len]
   */
  Tensor makeStreamingChunkMask(const Tensor& inputs_embeds, int32_t past_seen_tokens, const Tensor& streaming_tts_text_mask,
                                int32_t streaming_reserved_length = 300, int32_t streaming_text_chunk_size = 10) {
    // Only support batch size of 1 for inference
    MLLM_RT_ASSERT(inputs_embeds.shape()[0] == 1);

    auto dtype = inputs_embeds.dtype();
    auto device = inputs_embeds.device();
    auto seq_len = inputs_embeds.shape()[1];

    // Create mask tensor: [1, past_seen_tokens + seq_len]
    // Add `seq_len` to the past seen tokens to account for new tokens during generate
    auto mask_len = past_seen_tokens + seq_len;
    auto causal_mask = Tensor::zeros({1, mask_len}, dtype, device);

    // Get minimum value for this dtype (equivalent to -inf for masking)
    float min_val = -std::numeric_limits<float>::infinity();
    if (dtype == kFloat16) {
      min_val = -65504.0f;  // Approximate min value for float16
    }

    // Calculate the start of invisible text tokens
    // Using integer division and ceiling equivalent: (a + b - 1) / b
    int32_t chunk_index =
        (past_seen_tokens - streaming_reserved_length + cfg_.streaming_audio_chunk_size - 1) / cfg_.streaming_audio_chunk_size;
    int32_t text_tokens_from_chunks = chunk_index * streaming_text_chunk_size;
    int32_t clamped_text_tokens = std::min(text_tokens_from_chunks, streaming_reserved_length);

    int32_t invisible_text_tokens_start = clamped_text_tokens + 1 + cfg_.num_spk_embs * cfg_.use_speaker_embedding;

    int32_t invisible_text_tokens_end = streaming_reserved_length + 1 + cfg_.num_spk_embs * cfg_.use_speaker_embedding + 1;

    // Set invisible text tokens to min_val (effectively -inf)
    auto mask_ptr = causal_mask.ptr<float>();
    for (int32_t i = invisible_text_tokens_start; i < invisible_text_tokens_end && i < mask_len; ++i) { mask_ptr[i] = min_val; }

    // Mask padding positions in the text mask
    // Apply streaming_tts_text_mask to positions [0 : 1 + num_spk_embs * use_spk_emb + streaming_reserved_length + 1]
    int32_t text_mask_end = 1 + cfg_.num_spk_embs * cfg_.use_speaker_embedding + streaming_reserved_length + 1;
    auto text_mask_ptr = streaming_tts_text_mask.ptr<float>();
    int32_t text_mask_len = streaming_tts_text_mask.numel();

    for (int32_t i = 0; i < text_mask_end && i < mask_len && i < text_mask_len; ++i) {
      if (text_mask_ptr[i] == 0.0f) { mask_ptr[i] = min_val; }
    }

    // Add extra dimensions for batch and heads: [1, past_seen_tokens + seq_len] -> [1, 1, seq_len, past_seen_tokens + seq_len]
    // First unsqueeze to add head dimension: [1, 1, past_seen_tokens + seq_len]
    auto mask_with_head = causal_mask.unsqueeze(1);

    // Then unsqueeze to add seq dimension and repeat for seq_len: [1, 1, seq_len, past_seen_tokens + seq_len]
    auto final_mask = Tensor::zeros({1, 1, seq_len, mask_len}, dtype, device);
    auto final_mask_ptr = final_mask.ptr<float>();
    auto mask_with_head_ptr = mask_with_head.ptr<float>();

    // Copy the same mask for each sequence position
    for (int32_t s = 0; s < seq_len; ++s) {
      for (int32_t m = 0; m < mask_len; ++m) { final_mask_ptr[s * mask_len + m] = mask_with_head_ptr[m]; }
    }

    return final_mask;
  }

  /**
   * Greedy sampling: select token with highest probability
   */
  int64_t greedySample(const Tensor& scores) {
    // scores should be 1D tensor [vocab_size]
    MLLM_RT_ASSERT(scores.shape().size() == 1);
    MLLM_RT_ASSERT_EQ(scores.dtype(), kFloat32);

    auto scores_data = scores.ptr<float>();
    int vocab_size = scores.shape()[0];

    auto max_it = std::max_element(scores_data, scores_data + vocab_size);
    return std::distance(scores_data, max_it);
  }

  /**
   * Multinomial sampling from probability distribution
   */
  int64_t multinomialSample(const Tensor& scores) {
    // scores should be 1D tensor [vocab_size] with probabilities that sum to 1
    MLLM_RT_ASSERT(scores.shape().size() == 1);
    MLLM_RT_ASSERT_EQ(scores.dtype(), kFloat32);

    auto scores_data = scores.ptr<float>();
    int vocab_size = scores.shape()[0];

    // Compute cumulative distribution
    std::vector<float> cumulative_probs(vocab_size);
    std::partial_sum(scores_data, scores_data + vocab_size, cumulative_probs.begin());

    // Generate random number
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    float r = dis(gen);

    // Find the token
    auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), r);
    return std::distance(cumulative_probs.begin(), it);
  }

  /**
   * Top-K sampling for single token
   */
  int64_t sampleTopKSingle(const Tensor& scores, int k) {
    // scores should be 1D tensor [vocab_size]
    MLLM_RT_ASSERT(scores.shape().size() == 1);
    MLLM_RT_ASSERT_EQ(scores.dtype(), kFloat32);

    auto scores_data = scores.ptr<float>();
    int vocab_size = scores.shape()[0];

    // Create indices and partial sort
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&scores_data](int i1, int i2) { return scores_data[i1] > scores_data[i2]; });

    // Extract top-k probabilities
    std::vector<float> top_k_probs(k);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      top_k_probs[i] = scores_data[indices[i]];
      sum += top_k_probs[i];
    }

    // Normalize
    for (int i = 0; i < k; ++i) { top_k_probs[i] *= (1.0f / sum); }

    // Sample from top-k distribution
    std::mt19937 gen(Context::instance().getRandomSeed());
    std::discrete_distribution<> dist(top_k_probs.begin(), top_k_probs.end());
    int selected_idx = dist(gen);

    return indices[selected_idx];
  }

  /**
   * Top-P sampling for single token
   */
  int64_t sampleTopPSingle(const Tensor& scores, float p) {
    // scores should be 1D tensor [vocab_size]
    MLLM_RT_ASSERT(scores.shape().size() == 1);
    MLLM_RT_ASSERT_EQ(scores.dtype(), kFloat32);

    auto scores_data = scores.ptr<float>();
    int vocab_size = scores.shape()[0];

    // Create indices and sort by probability
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scores_data](int i1, int i2) { return scores_data[i1] > scores_data[i2]; });

    // Find tokens in top-p nucleus
    std::vector<float> top_probs;
    float cumulative_prob = 0.0f;
    int i = 0;
    for (; i < vocab_size && cumulative_prob < p; ++i) {
      top_probs.push_back(scores_data[indices[i]]);
      cumulative_prob += scores_data[indices[i]];
    }

    // Normalize probabilities
    float sum = std::accumulate(top_probs.begin(), top_probs.end(), 0.0f);
    for (float& prob : top_probs) { prob *= (1.0f / sum); }

    // Sample from top-p distribution
    std::mt19937 gen(Context::instance().getRandomSeed());
    std::discrete_distribution<> dist(top_probs.begin(), top_probs.end());
    int selected_idx = dist(gen);

    return indices[selected_idx];
  }
};
}  // namespace mllm::models::chattts
