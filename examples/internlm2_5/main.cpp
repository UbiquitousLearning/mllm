#include <iostream>
#include <fmt/core.h>
#include <chrono>
#include <mllm/mllm.hpp>
#include <mllm/models/internlm2/modeling_internlm2.hpp>
#include <mllm/models/internlm2/tokenization_internlm2.hpp>

using mllm::Argparse;
using Clock = std::chrono::high_resolution_clock;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer JSON path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  Argparse::parse(argc, argv);

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.get() == "v2") file_version = mllm::ModelFileVersion::kV2;
  if (help.isSet()) { Argparse::printHelp(); mllm::shutdownContext(); return 0; }

  auto cfg = mllm::models::internlm2::InternLM2Config(config_path.get());
  auto tokenizer = mllm::models::internlm2::InternLM2Tokenizer(tokenizer_path.get());
  auto model = mllm::models::internlm2::InternLM2ForCausalLM(cfg);
  auto params = mllm::load(model_path.get(), file_version);
  model.load(params);

  fmt::print("\n{:*^60}\n", " InternLM2.5 1.5B CLI ");

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);
  if (!(prompt_text == "exit" || prompt_text == "quit")) {
    fmt::print("🔄 Processing...\n");
    mllm::models::internlm2::InternLM2Message prompt{prompt_text};
    auto inputs = tokenizer.convertMessage(prompt);

    auto t0 = Clock::now();
    fmt::print("\n🤖 Response: ");

    int tok_count = 0;
    auto t_first = t0;
    for (auto& step : model.chat(inputs)) {
      if (tok_count == 0) t_first = Clock::now();
      auto token = tokenizer.detokenize(step.cur_token_id);
      std::wcout << token << std::flush;
      tok_count++;
      if (tok_count >= 30) break;
    }

    auto t_end = Clock::now();
    auto prefill_ms = std::chrono::duration<double, std::milli>(t_first - t0).count();
    auto total_s = std::chrono::duration<double>(t_end - t0).count();

    fmt::print("\n{}\n", std::string(60, '-'));
    fmt::print("⏱ Prefill: {:.0f}ms | ", prefill_ms);
    if (tok_count > 1) {
      auto decode_s = std::max(0.001, total_s - prefill_ms/1000.0);
      fmt::print("Decode: {:.1f} tok/s ({} tokens in {:.1f}s)\n",
                 (tok_count-1)/decode_s, tok_count, total_s);
    }
  }
  mllm::print("\n");
  mllm::memoryReport();
})
