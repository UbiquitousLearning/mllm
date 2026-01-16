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
  auto& temperature = Argparse::add<float>("--temperature").help("Temperature").def(0.8f);
  auto& ar_len = Argparse::add<int>("--ar_len").help("Autoregressive length (chunk size)").def(128);

  Argparse::parse(argc, argv);

  mllm::initQnnBackend(model_path.get());

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  RunnerConfig config;
  config.model_path = model_path.get();
  config.temperature = temperature.get();
  config.num_layers = qwen3_cfg.num_hidden_layers;
  config.num_heads = qwen3_cfg.num_attention_heads;
  config.head_dim = qwen3_cfg.head_dim;
  config.vocab_size = qwen3_cfg.vocab_size;
  config.context_len = 1024;
  config.ar_len = ar_len.get();

  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);

  auto input_tensor = tokenizer.convertMessage({.prompt = prompt_text});

  Runner runner(config, &tokenizer);
  if (!runner.load()) {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  std::vector<uint64_t> prompt_tokens;
  auto sequence = input_tensor["sequence"];
  int64_t* ptr = sequence.ptr<int64_t>();
  for (int i = 0; i < sequence.shape()[1]; ++i) { prompt_tokens.push_back((uint64_t)ptr[i]); }

  runner.generate(prompt_tokens, config.context_len, [](const std::string& token) { std::cout << token << std::flush; });
  std::cout << "\n";

  return 0;
});