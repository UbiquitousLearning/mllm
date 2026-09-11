// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <cstdio>
#include <fstream>
#include <iostream>
#include "mllm/mllm.hpp"
#include "mllm/models/spark2_5/modeling_spark2_5.hpp"
#include "mllm/models/spark2_5/tokenization_spark2_5.hpp"
using mllm::Argparse;
MLLM_MAIN({
  auto engine_args = mllm::engineArgAttach();
  auto& model_path = Argparse::add<std::string>("-m|--model_path").required(true).help("Converted V2 model");
  auto& config_path = Argparse::add<std::string>("-c|--config_path").required(true).help("Spark runtime config");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").required(true).help("Official tokenizer.json");
  auto& prompt = Argparse::add<std::string>("-p|--prompt").required(true).help("Prompt text");
  auto& system = Argparse::add<std::string>("--system").help("Optional system message");
  auto& thinking = Argparse::add<std::string>("--enable_thinking").help("Enable reasoning (default true)");
  auto& count = Argparse::add<int>("-g|--max_new_tokens").help("Maximum generated tokens (default 128)");
  auto& print_ids = Argparse::add<bool>("--print_token_ids").help("Write generated token IDs to stderr");
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      Argparse::printHelp();
      return 0;
    }
  Argparse::parse(argc, argv);
  mllm::Context::instance().setCpuOpThreads(1);
  mllm::configEngineWithArgs(engine_args);
  try {
    mllm::models::spark2_5::SparkConfig config(config_path.get());
    mllm::models::spark2_5::SparkTokenizer tokenizer(tokenizer_path.get());
    const int limit = count.isSet() ? count.get() : 128;
    if (limit <= 0) throw std::invalid_argument("max_new_tokens must be positive");
    const auto thinking_value = thinking.isSet() ? thinking.get() : "true";
    if (thinking_value != "true" && thinking_value != "false")
      throw std::invalid_argument("enable_thinking must be true or false");
    const auto inputs = tokenizer.convertMessage({prompt.get(), system.isSet() ? system.get() : "", thinking_value == "true"});
    const int prompt_tokens = inputs.at("sequence").shape()[1];
    if (limit > config.max_cache_length || prompt_tokens > config.max_cache_length - limit + 1)
      throw std::invalid_argument("Prompt and generation exceed context capacity");
    auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
    mllm::models::spark2_5::validateModelConfigMatch(config, params);
    mllm::models::spark2_5::SparkForCausalLM model(config);
    model.load(params);
    model.resetState();
    mllm::preprocessor::StreamingUtf8Decoder decoder;
    int generated = 0;
    model.streamGenerate(inputs, {{"max_length", mllm::AnyValue(int(limit))}, {"do_sample", mllm::AnyValue(false)}},
                         [&](int64_t id) {
                           ++generated;
                           if (print_ids.isSet() && print_ids.get()) fmt::print(stderr, "TOKEN_ID:{}\n", id);
                           fmt::print("{}", decoder.append(tokenizer.detokenizeBytes(id)));
                           std::fflush(stdout);
                         });
    fmt::print("{}\n", decoder.finish());
    fmt::print(stderr, "Prompt tokens: {}; generated tokens: {}\n", prompt_tokens, generated);
  } catch (const std::exception& e) {
    fmt::print(stderr, "Spark: {}\n", e.what());
    return 1;
  }
  return 0;
})
