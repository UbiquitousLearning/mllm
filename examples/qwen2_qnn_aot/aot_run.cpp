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
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Model path").def("qwen2_qnn.mllm");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Config path").required(true);
  auto& ar_len = Argparse::add<int>("--ar_len").help("Autoregressive length (chunk size)").def(128);

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  mllm::initQnnBackend(model_path.get());

  auto qwen2_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  RunnerConfig config;
  config.num_layers = qwen2_cfg.num_hidden_layers;
  config.num_heads = qwen2_cfg.num_attention_heads;
  config.head_dim = qwen2_cfg.head_dim;
  config.vocab_size = qwen2_cfg.vocab_size;
  config.context_len = 1024;
  config.ar_len = ar_len.get();

  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  auto input_tensor = tokenizer.convertMessage({.prompt = "hello"});

  input_tensor["sequence"] = mllm::Tensor::arange(0, 256, 1, mllm::kInt64, mllm::kCPU).view({1, -1});

  // DBG:
  mllm::print(input_tensor["sequence"].shape());
  mllm::print(input_tensor["sequence"]);

  Runner runner(config, &tokenizer);
  if (!runner.load()) {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  runner.generate(input_tensor["sequence"], 128, [](const std::string& token) { std::cout << token << std::flush; }, true);
  std::cout << "\n";

  return 0;
});
