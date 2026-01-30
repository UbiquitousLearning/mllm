#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <string>
#include "mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Model path").def("qwen3_qnn.mllm");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Config path").required(true);
  auto& ar_len = Argparse::add<int>("--ar_len").help("Autoregressive length (chunk size)").def(128);

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  mllm::initQnnBackend(model_path.get());

  auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  RunnerConfig config;
  config.num_layers = qwen3_cfg.num_hidden_layers;
  config.num_heads = qwen3_cfg.num_key_value_heads;
  config.head_dim = qwen3_cfg.head_dim;
  config.vocab_size = qwen3_cfg.vocab_size;
  config.context_len = 1024;
  config.ar_len = ar_len.get();

  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);

  auto input_tensor = tokenizer.convertMessage({.prompt = prompt_text});

  // DBG:
  mllm::print(input_tensor["sequence"].shape());
  mllm::print(input_tensor["sequence"]);

  Runner runner(config, &tokenizer);
  if (!runner.load()) {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  runner.generate(input_tensor["sequence"], config.context_len,
                  [](const std::string& token) { std::cout << token << std::flush; });
  std::cout << "\n";

  return 0;
});
