#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen3_5/modeling_qwen3_5.hpp>
#include <mllm/models/qwen3_5/tokenization_qwen3_5.hpp>
#include <mllm/utils/AnyValue.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

  Argparse::parse(argc, argv);

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

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
    auto cfg = mllm::models::qwen3_5::Qwen3_5Config(config_path.get());
    auto tokenizer = mllm::models::qwen3_5::Qwen3_5Tokenizer(tokenizer_path.get());
    auto model = mllm::models::qwen3_5::Qwen3_5ForCausalLM(cfg);

    fmt::print("Qwen3.5 0.8B: {} layers ({} full attention + {} GDN)\n",
               cfg.num_hidden_layers, cfg.numFullAttentionLayers(), cfg.numGDNLayers());

    auto param = mllm::load(model_path.get(), file_version);
    model.load(param);

    fmt::print("\n{:*^60}\n", " Qwen3.5 Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text;

    fmt::print("Prompt text (or 'exit/quit'): ");
    std::getline(std::cin, prompt_text);

    try {
      fmt::print("Processing...\n");
      auto inputs = tokenizer.convertMessage({.prompt = prompt_text});

      fmt::print("\nResponse: ");

      for (auto& step : model.chat(inputs)) { std::wcout << tokenizer.detokenize(step.cur_token_id) << std::flush; }

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\nError: {}\n{}\n", e.what(), std::string(60, '-')); }

    model.perfSummary();
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3_5.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
})
