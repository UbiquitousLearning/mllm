// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen2_5omni/configuration_qwen2_5omni.hpp>
#include <mllm/models/qwen2_5omni/modeling_qwen2_5omni.hpp>
#include <mllm/models/qwen2_5omni/tokenization_qwen2_5omni.hpp>

using mllm::Argparse;

//MLLM_MAIN({
int main(int argc, char** argv) {
  ::mllm::__setup_signal_handler();
  ::mllm::initializeContext();

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
    auto qwen2_5omni_tokenizer =
        mllm::models::qwen2_5omni::Qwen2_5OmniTokenizer(tokenizer_path.get(), qwen2_5omni_cfg.visual_spatial_merge_size);
    auto qwen2_5omni = mllm::models::qwen2_5omni::Qwen2_5OmniForCausalLM(qwen2_5omni_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    qwen2_5omni.thinker_.load(param);

    fmt::print("\n{:*^60}\n", " Qwen2.5-Omni Image CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string image_path;
    std::string prompt_text;

    fmt::print("Image path (or 'exit/quit'): ");
    image_path = "../../rsc/pics.jpg";
    //std::getline(std::cin, image_path);
    if (image_path == "exit" || image_path == "quit") { return 0; }

    fmt::print("Prompt text: ");
    prompt_text = "描述图片中物体";
    //std::getline(std::cin, prompt_text);

    try {
      fmt::print("Processing...\n");
      auto inputs = qwen2_5omni_tokenizer.convertVisionMessage({.prompt = prompt_text, .img_file_path = image_path});

      fmt::print("\nResponse: ");
      qwen2_5omni.streamGenerate(inputs,
                                 {
                                     {"do_sample", mllm::AnyValue(false)},
                                     {"max_length", mllm::AnyValue(qwen2_5omni_cfg.max_cache_length)},
                                 },
                                 [&](int64_t token_id) {
                                   auto str = qwen2_5omni_tokenizer.detokenize(token_id);
                                   std::wcout << str << std::flush;
                                 });

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\nError: {}\n{}\n", e.what(), std::string(60, '-')); }

    qwen2_5omni.perfSummary();
  }

  mllm::print("\n");
  mllm::memoryReport();

  ::mllm::shutdownContext();
  return 0;
}
