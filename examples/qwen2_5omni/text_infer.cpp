// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen2_5omni/configuration_qwen2_5omni.hpp>
#include <mllm/models/qwen2_5omni/modeling_qwen2_5omni.hpp>
#include <mllm/models/qwen2_5omni/tokenization_qwen2_5omni.hpp>

using mllm::Argparse;

MLLM_MAIN({
  mllm::Logger::level() = mllm::LogLevel::kError;
  
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

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

  {
    auto qwen2_5omni_cfg = mllm::models::qwen2_5omni::Qwen2_5OmniConfig(config_path.get());
    auto qwen2_5omni_tokenizer = mllm::models::qwen2_5omni::Qwen2_5OmniTokenizer(tokenizer_path.get());
    auto qwen2_5omni = mllm::models::qwen2_5omni::Qwen2_5OmniForCausalLM(qwen2_5omni_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    qwen2_5omni.thinker_.load(param);

    fmt::print("\n{:*^60}\n", " Qwen2.5-Omni Text CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text;

    fmt::print("💬 Prompt text (or 'exit/quit'): ");
    std::getline(std::cin, prompt_text);

    if (prompt_text == "exit" || prompt_text == "quit") { return 0; }

    try {
      fmt::print("🔄 Processing...\n");
      auto inputs = qwen2_5omni_tokenizer.convertMessage({.prompt = prompt_text});

      fmt::print("\n🤖 Response: ");
      for (auto& step : qwen2_5omni.chat(inputs)) {
        std::wcout << qwen2_5omni_tokenizer.detokenize(step.cur_token_id) << std::flush;
      }

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }

    qwen2_5omni.perfSummary();
  }

  mllm::print("\n");
  mllm::memoryReport();
})
