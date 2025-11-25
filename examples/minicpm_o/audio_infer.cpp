// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <iostream>
#include <fmt/core.h>
#include "mllm/mllm.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/modeling_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/modeling_resampler.hpp"
#include "mllm/models/minicpm_o2_6/modeling_siglip.hpp"
#include "mllm/models/minicpm_o2_6/tokenization_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/audio_preprocessor_minicpmo.hpp"

using mllm::Argparse;

MLLM_MAIN({
  mllm::Logger::level() = mllm::LogLevel::kInfo;

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& audio_path = Argparse::add<std::string>("-a|--audio_path").help("Audio file path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Text prompt (default: Please describe the audio.)");

  /*
    FOR RUN (MacOS Apple Silicon):
      python task.py tasks/build_osx_apple_silicon.yaml
      cd build-osx/bin
      ./main_audio -m ../../models/minicpm-o-2_6.mllm -mv v1 \
        -t ../../tokenizer/MiniCPM-o-2_6/tokenizer.json \
        -c ../../examples/minicpm_o/config_minicpm_o.json \
        -a ../../models/recognize.wav \
        -p "What is being said in this audio?"
  */

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  }

  fmt::print("\n{:*^80}\n", " MiniCPM-o Audio Test ");
  fmt::print("Loading model and configuration...\n");

  auto minicpmo_cfg = mllm::models::minicpmo::MiniCPMOConfig(config_path.get());
  auto minicpmo_tokenizer = mllm::models::minicpmo::MiniCPMOTokenizer(tokenizer_path.get());
  auto minicpmo = mllm::models::minicpmo::MiniCPMOForCausalLM(minicpmo_cfg);

  auto param = mllm::load(model_path.get(), file_version);

  minicpmo.llm_.llm.load(param);
  minicpmo.vpm_.load(param);
  minicpmo.resampler_.load(param);
  minicpmo.apm_.load(param);
  minicpmo.audio_projection_layer_.load(param);

  fmt::print("Model loaded successfully!\n");

  mllm::models::minicpmo::MiniCPMOMessage message;
  message.prompt = prompt.isSet() ? prompt.get() : "Please describe the audio.";
  message.audio_file_path = audio_path.get();

  auto inputs = minicpmo_tokenizer.convertMessage(message);

  fmt::print("\n{:*^80}\n", " Generating Response ");
  fmt::print("Response: ");

  int token_count = 0;
  int max_tokens = 200;  // Limit for testing

  for (auto& step : minicpmo.chat(inputs)) {
    auto token_str = minicpmo_tokenizer.detokenize(step.cur_token_id);
    std::wcout << token_str << std::flush;
    token_count++;
    if (token_count >= max_tokens) {
      fmt::print("\n[Reached max token limit: {}]\n", max_tokens);
      break;
    }
  }

  fmt::print("\n{:*^80}\n", " Test Complete ");
  fmt::print("Generated {} tokens\n", token_count);

  mllm::memoryReport();
  mllm::shutdownContext();
  return 0;
})
