#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <string>
#include "mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp"
#include "configuration_llama3.hpp"
#include "mllm/models/llama/tokenization_llama.hpp"

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Model path").def("llama_qnn.mllm");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Config path").required(true);
  auto& ar_len = Argparse::add<int>("--ar_len").help("Autoregressive length (chunk size)").def(128);
  // auto& seq_len = Argparse::add<int>("--seq_len").help("Input sequence length").def(800);
  // auto& gen_len = Argparse::add<int>("--gen_len").help("Generate token length").def(32);

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  mllm::initQnnBackend(model_path.get());

  auto llama_cfg = mllm::models::llama3::Llama3Config(config_path.get());

  RunnerConfig config;
  config.num_layers = llama_cfg.num_hidden_layers;
  config.num_heads = llama_cfg.num_attention_heads;
  config.head_dim = llama_cfg.head_dim;
  config.vocab_size = llama_cfg.vocab_size;
  config.context_len = 1024;
  config.ar_len = ar_len.get();
  config.type = "llama3";

  // Note: Using Qwen3 tokenizer as a placeholder.
  // For production use, you should implement a Llama3Tokenizer or use
  // the appropriate tokenizer for your model.
  auto tokenizer = mllm::models::llama::LlamaTokenizer(tokenizer_path.get());

  // auto input_tensor = tokenizer.convertMessage({{
  //     .role = "user",
  //     .content = "hello",
  // }});

  // input_tensor["sequence"] = mllm::Tensor::arange(0, seq_len.get(), 1, mllm::kInt64, mllm::kCPU).view({1, -1});

  // // DBG:
  // mllm::print(input_tensor["sequence"].shape());
  // mllm::print(input_tensor["sequence"]);

  // Runner runner(config, &tokenizer);
  // if (!runner.load()) {
  //   std::cerr << "Failed to load model\n";
  //   return 1;
  // }

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);

  auto input_tensor = tokenizer.convertMessage({{.role = "user", .content = prompt_text}});

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
