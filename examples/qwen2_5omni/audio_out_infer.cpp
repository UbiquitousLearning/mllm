// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen2_5omni/configuration_qwen2_5omni.hpp>
#include <mllm/models/qwen2_5omni/modeling_qwen2_5omni.hpp>
#include <mllm/models/qwen2_5omni/tokenization_qwen2_5omni.hpp>
#include "wenet_audio/wav.h"

using mllm::Argparse;

MLLM_MAIN({
  mllm::Logger::level() = mllm::LogLevel::kError;

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& spk_dict_path = Argparse::add<std::string>("-s|--spk_dict_path").help("Speaker json path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Prompt text").def("");
  auto& image_path = Argparse::add<std::string>("-i|--image_path").help("Image path").def("");
  auto& audio_path = Argparse::add<std::string>("-a|--audio_path").help("Audio path").def("");
  auto& speaker = Argparse::add<std::string>("-sp|--speaker").help("Speaker name (default: first entry)").def("");
  auto& output_path = Argparse::add<std::string>("-o|--output_path").help("Output wav path").def("./qwen2_5omni.wav");

  Argparse::parse(argc, argv);

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  }

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  if (!image_path.get().empty() && !audio_path.get().empty()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Only one of --image_path or --audio_path can be set.");
  }

  auto qwen_cfg = mllm::models::qwen2_5omni::Qwen2_5OmniConfig(config_path.get());
  auto qwen_tokenizer = mllm::models::qwen2_5omni::Qwen2_5OmniTokenizer(tokenizer_path.get());
  auto qwen_omni = mllm::models::qwen2_5omni::Qwen2_5OmniForConditionalGeneration(qwen_cfg);

  auto param = mllm::load(model_path.get(), file_version);
  qwen_omni.load(param);
  qwen_omni.loadSpeakers(spk_dict_path.get());

  std::string prompt_text = prompt.get();
  if (prompt_text.empty()) {
    fmt::print("Prompt text: ");
    std::getline(std::cin, prompt_text);
    if (prompt_text.empty()) { prompt_text = "Please respond."; }
  }

  mllm::models::ARGenerationOutputPast inputs;
  if (!image_path.get().empty()) {
    inputs = qwen_tokenizer.convertVisionMessage({.prompt = prompt_text, .img_file_path = image_path.get()});
  } else if (!audio_path.get().empty()) {
    inputs = qwen_tokenizer.convertAudioMessage({.prompt = prompt_text, .audio_file_path = audio_path.get()});
  } else {
    inputs = qwen_tokenizer.convertMessage({.prompt = prompt_text});
  }

  mllm::models::qwen2_5omni::Qwen2_5OmniAudioGenerationConfig gen_cfg;
  auto output = qwen_omni.generateAudio(inputs, gen_cfg, speaker.get());

  auto input_len = inputs["sequence"].shape()[1];
  auto total_len = output.sequences.shape()[1];
  fmt::print("\nResponse: ");
  for (int i = input_len; i < total_len; ++i) {
    std::wcout << qwen_tokenizer.detokenize(output.sequences.at<mllm::mllm_int64_t>({0, i})) << std::flush;
  }
  fmt::print("\n");

  auto wav = output.wav * 32767.0f;
  wenet::WavWriter wav_writer(wav.ptr<float>(), wav.shape().back(), 1, 24000, 16);
  wav_writer.Write(output_path.get());

  fmt::print("Saved audio to {}\n", output_path.get());

  qwen_omni.thinker_.perfSummary();

  mllm::print("\n");
  mllm::memoryReport();
})
