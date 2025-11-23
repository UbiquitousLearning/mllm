// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <vector>
#include "fmt/base.h"
#include "mllm/core/Tensor.hpp"
#include "mllm/models/minicpm_o2_6/configuration_chattts.hpp"
#include "mllm/models/minicpm_o2_6/modeling_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/tokenization_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/streaming_generation.hpp"
#include "wenet_audio/wav.h"

using namespace mllm;  // NOLINT

MLLM_MAIN({
  mllm::Logger::level() = mllm::LogLevel::kError;

  std::string model_path = "path/to/your/minicpm-o-2_6.mllm";
  std::string tokenizer_path = "path/to/your//tokenizer.json";
  std::string config_path = "../../examples/minicpm_o/config_minicpm_o.json";
  std::string chattts_config_path = "../../examples/minicpm_o/config_chattts.json";
  std::string chattts_tokenizer_path = "path/to/your//tokenizer.json";
  std::string vocos_model_path = "path/to/your/vocos.mllm";
  std::string model_version = "v1";

  auto config = models::minicpmo::MiniCPMOConfig(config_path);
  auto model = models::minicpmo::MiniCPMOForCausalLM(config);
  auto minicpmo_tokenizer = models::minicpmo::MiniCPMOTokenizer(tokenizer_path);
  models::chattts::ChatTTSTokenizer chattts_tokenizer(chattts_tokenizer_path);

  auto param = load(model_path, ModelFileVersion::kV1);
  model.llm_.llm.load(param);
  model.vpm_.load(param);
  model.resampler_.load(param);
  model.apm_.load(param);
  model.audio_projection_layer_.load(param);

  auto chattts_config = models::chattts::ChatTTSConfig(chattts_config_path);
  model.init_tts_module(chattts_config);
  model.tts_model_.load(param);

  auto vocos_model = models::vocos::Vocos("", 512, 1536, 8, 1024, 256, 100, "center");
  vocos_model.from_pretrained(vocos_model_path);
  model.vocos_model_ = &vocos_model;

  // Change Your Inputs Here
  std::string image_path = "path/to/your/pics.jpg";
  std::string audio_path = "path/to/your/describe.wav";
  std::string prompt_text = "根据我的图片和语音，完成任务";

  mllm::models::minicpmo::MiniCPMOMessage message;
  message.prompt = prompt_text;
  message.img_file_path = image_path;
  message.audio_file_path = audio_path;

  auto inputs = minicpmo_tokenizer.convertMessage(message);

  print("Models loaded successfully!");

  auto prefill_out = model.forward(inputs, {});

  auto sample = model.sampleGreedy(prefill_out["sequence"]);
  auto token_str = minicpmo_tokenizer.detokenize(sample);
  std::wcout << token_str << std::flush;

  std::string generate_prompt = "<|im_end|>\n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>";
  auto gen_ids = minicpmo_tokenizer.convert2Ids(minicpmo_tokenizer.tokenize(generate_prompt));

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

  // int token_count = 0;

  // for (auto& step : model.chat(inputs)) {
  //   auto token_str = minicpmo_tokenizer.detokenize(step.cur_token_id);
  //   std::wcout << token_str << std::flush;

  //   token_count++;
  //   if (token_count >= 200) break;
  // }

  print("\n");

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("minicpm4.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
})
