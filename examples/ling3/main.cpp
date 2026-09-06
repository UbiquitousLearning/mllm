// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <fmt/core.h>

#include <mllm/engine/Context.hpp>
#include <mllm/mllm.hpp>
#include <mllm/models/ling3/modeling_ling3.hpp>
#include <mllm/models/ling3/tokenization_ling3.hpp>
#include <mllm/utils/AnyValue.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using mllm::Argparse;

namespace {

std::string readPromptFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) { throw std::invalid_argument("unable to read prompt_file: " + path); }
  std::string text{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) { text.pop_back(); }
  if (text.empty()) { throw std::invalid_argument("prompt_file must not be empty: " + path); }
  return text;
}

}  // namespace

MLLM_MAIN({
  auto engine_args = mllm::engineArgAttach();
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("MLLM V2 model path").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Official tokenizer.json").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Ling-3 mobile runtime config").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Run one prompt non-interactively").required(false);
  auto& prompt_file = Argparse::add<std::string>("--prompt_file").help("Read a prompt from a UTF-8 file").required(false);
  auto& max_new_tokens =
      Argparse::add<int>("-g|--max_new_tokens").help("Maximum generated tokens (default: 8)").required(false);
  auto& min_new_tokens = Argparse::add<int>("--min_new_tokens").help("Suppress EOS until this many tokens").required(false);
  auto& disable_thinking =
      Argparse::add<bool>("--disable_thinking").help("Use the official thinking-off chat template").required(false);
  auto& print_token_ids = Argparse::add<bool>("--print_token_ids").help("Print generated token IDs to stderr").required(false);

  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) == "-h" || std::string(argv[index]) == "--help") {
      Argparse::printHelp();
      return 0;
    }
  }
  Argparse::parse(argc, argv);
  mllm::configEngineWithArgs(engine_args);
  (void)help;

  const auto config = mllm::models::ling3::Ling3Config(config_path.get());
  int generation_limit = max_new_tokens.isSet() ? max_new_tokens.get() : 8;
  int minimum_generation = min_new_tokens.isSet() ? min_new_tokens.get() : 0;
  if (generation_limit <= 0 || generation_limit > config.max_cache_length || minimum_generation < 0
      || minimum_generation > generation_limit) {
    throw std::invalid_argument("generation lengths must satisfy 0 <= min_new_tokens <= max_new_tokens <= max_cache_length");
  }
  if (prompt.isSet() && prompt_file.isSet()) { throw std::invalid_argument("prompt and prompt_file are mutually exclusive"); }

  std::string configured_prompt;
  if (prompt_file.isSet()) {
    configured_prompt = readPromptFile(prompt_file.get());
  } else if (prompt.isSet()) {
    configured_prompt = prompt.get();
  }

  const auto parameters = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  mllm::models::ling3::validateLing3ModelConfigMatch(config, parameters);
  auto tokenizer = mllm::models::ling3::Ling3Tokenizer(tokenizer_path.get());
  auto model = mllm::models::ling3::Ling3ForCausalLM(config);
  model.load(parameters);
  fmt::print("Ling-3.0-tiny: {} layers ({} MLA + {} KDA), CPU threads={}\n", config.num_hidden_layers,
             config.numFullAttentionLayers(), config.numKDALayers(), mllm::Context::instance().getCpuOpThreads());

  int exit_code = 0;
  while (true) {
    std::string prompt_text = configured_prompt;
    if (!prompt.isSet() && !prompt_file.isSet()) {
      fmt::print("Prompt text (or 'exit/quit'): ");
      if (!std::getline(std::cin, prompt_text) || prompt_text == "exit" || prompt_text == "quit") { break; }
    }
    if (prompt_text.empty()) {
      if (prompt.isSet() || prompt_file.isSet()) { throw std::invalid_argument("prompt must not be empty"); }
      continue;
    }
    try {
      auto input = tokenizer.convertMessage({.prompt = prompt_text,
                                             .system_prompt = "",
                                             .enable_thinking = !(disable_thinking.isSet() && disable_thinking.get())});
      const int prompt_tokens = input.at("sequence").shape()[1];
      if (prompt_tokens + generation_limit - 1 > config.max_cache_length) {
        throw std::invalid_argument("prompt plus generation exceeds max_cache_length");
      }
      model.resetState();
      fmt::print("LING3_RUN_START prompt_tokens={} max_new_tokens={} min_new_tokens={}\nResponse: ", prompt_tokens,
                 generation_limit, minimum_generation);
      int generated_tokens = 0;
      for (const auto& step : model.chat(input, {{"max_length", mllm::AnyValue(generation_limit)},
                                                 {"min_new_tokens", mllm::AnyValue(minimum_generation)},
                                                 {"do_sample", mllm::AnyValue(false)}})) {
        if (print_token_ids.isSet() && print_token_ids.get()) { fmt::print(stderr, "LING3_TOKEN_ID:{}\n", step.cur_token_id); }
        fmt::print("{}", tokenizer.detokenizeBytes(step.cur_token_id));
        std::fflush(stdout);
        ++generated_tokens;
      }
      fmt::print("\nLING3_RUN_OK prompt_tokens={} generated_tokens={}\n", prompt_tokens, generated_tokens);
    } catch (const std::exception& error) {
      fmt::print(stderr, "LING3_RUN_ERROR:{}\n", error.what());
      exit_code = 1;
    }
    if (prompt.isSet() || prompt_file.isSet()) { break; }
  }

  model.perfSummary();
  mllm::memoryReport();
  return exit_code;
})
