// Copyright (c) MLLM Team.
// Licensed under the MIT License.

/**
 * @file demo_minicpmo_streaming.cpp
 * @brief Demo for MiniCPM-o streaming multimodal generation
 *
 * This example demonstrates how to use MiniCPM-o model for streaming
 * text and audio generation. It shows:
 * 1. Loading the model and TTS components
 * 2. Preparing input with text/image/audio
 * 3. Streaming generation with real-time audio output
 */

#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <vector>
#include "fmt/base.h"
#include "mllm/core/Tensor.hpp"
#include "mllm/models/minicpm_o2_6/modeling_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/tokenization_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/streaming_generation.hpp"
#include "wenet_audio/wav.h"

using namespace mllm;  // NOLINT

MLLM_MAIN({
  mllm::Logger::level() = mllm::LogLevel::kError;
  // ============================================================================
  // Configuration
  // ============================================================================
  std::string model_path = "<path-to-model>/minicpm-o2_6-q40.mllm";
  std::string tokenizer_path = "<path-to-model>/cpmo-tokenizer.json";
  std::string chattts_tokenizer_path = "<path-to-model>/tokenizer.json";
  std::string vocos_model_path = "<path-to-model>/vocos.mllm";

  // ============================================================================
  // Load Models
  // ============================================================================
  print("Loading MiniCPM-o model...");

  auto config = models::minicpmo::MiniCPMOConfig("<path-to-model>/config_minicpm_o.json");
  auto model = models::minicpmo::MiniCPMOForCausalLM(config);

  auto param = load(model_path, ModelFileVersion::kV1);
  model.llm_.llm.load(param);
  model.vpm_.load(param);
  model.resampler_.load(param);

  print("Loading TTS components...");

  // Load ChatTTS
  auto chattts_config = models::chattts::ChatTTSConfig("<path-to-config>/minicpm_o/config_chattts.json");
  model.init_tts_module(chattts_config);
  model.tts_model_.load(param);

  // Load Vocos
  auto vocos_model = models::vocos::Vocos("", 512, 1536, 8, 1024, 256, 100, "center");
  vocos_model.from_pretrained(vocos_model_path);
  model.vocos_model_ = &vocos_model;

  print("Models loaded successfully!");

  // ============================================================================
  // Prepare Input
  // ============================================================================
  auto minicpmo_tokenizer = models::minicpmo::MiniCPMOTokenizer(tokenizer_path);

  std::string image_path = "<path-to-image>";
  mllm::models::minicpmo::MiniCPMOMessage message;
  message.prompt = "What is this";  // FIXME: when prompt is empty, the output is wired (accuracy related)
  message.img_file_path = image_path;

  fmt::print("Processing...\n");

  auto inputs = minicpmo_tokenizer.convertMessage(message);

  auto prefill_out = model.forward(inputs, {});

  auto sample = model.sampleGreedy(prefill_out["sequence"]);
  auto token_str = minicpmo_tokenizer.detokenize(sample);
  std::wcout << token_str << std::flush;

  fmt::print("\nPerformed image prefilling\n");

  // ============================================================================
  // Create Streaming Generator
  // ============================================================================
  fmt::print("\n=== Starting Streaming Audio Generation ===\n");

  std::string generate_prompt = "<|im_end|>\n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>";
  auto gen_ids = minicpmo_tokenizer.convert2Ids(minicpmo_tokenizer.tokenize(generate_prompt));

  mllm::models::chattts::ChatTTSTokenizer chattts_tokenizer(chattts_tokenizer_path);

  models::minicpmo::StreamingGenerationConfig stream_config;
  stream_config.generate_audio = true;
  stream_config.output_chunk_size = 25;
  stream_config.max_new_tokens = 100;
  stream_config.force_no_stop = false;
  stream_config.top_p = 0.7f;
  stream_config.top_k = 20;
  stream_config.sampling = false;
  stream_config.tts_temperature = {0.1f, 0.3f, 0.1f, 0.3f};

  auto streaming_gen = models::minicpmo::StreamingGenerator(gen_ids, Tensor::nil(), model, minicpmo_tokenizer,
                                                            &chattts_tokenizer, stream_config, chattts_config);

  // ============================================================================
  // Stream Outputs
  // ============================================================================
  int chunk_count = 0;
  std::vector<Tensor> audio_chunks;
  for (auto& output : streaming_gen) {
    fmt::print("\n--- Chunk {} ---\n", chunk_count);
    ++chunk_count;
    fmt::print("Text: {}\n", output.text);

    if (output.audio_wav && !output.audio_wav.value().isNil()) {
      auto& audio = output.audio_wav.value();
      fmt::print("Audio: [{} samples @ {}Hz]\n", audio.shape()[1], output.sampling_rate);
      audio_chunks.emplace_back(audio);
    }

    if (output.finished) {
      fmt::print("\n=== Generation Finished ===\n");
      break;
    }
  }

  Tensor audio_output = nn::functional::concat(audio_chunks, -1);

  print("Final audio shape:", audio_output.shape(), audio_output);
  audio_output = audio_output * 32767;

  wenet::WavWriter wav_writer2(audio_output.ptr<float>(), audio_output.shape().back(), 1, 24000, 16);
  wav_writer2.Write("./omni.wav");
});
