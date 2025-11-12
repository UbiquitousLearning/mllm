#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/internlm2/modeling_internlm2.hpp>
#include <mllm/models/internlm2/tokenization_internlm2.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer JSON path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

  Argparse::parse(argc, argv);

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.get() == "v2") { file_version = mllm::ModelFileVersion::kV2; }

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  auto cfg = mllm::models::internlm2::InternLM2Config(config_path.get());
  auto tokenizer = mllm::models::internlm2::InternLM2Tokenizer(tokenizer_path.get());
  auto model = mllm::models::internlm2::InternLM2ForCausalLM(cfg);

  auto params = mllm::load(model_path.get(), file_version);
  model.load(params);

  fmt::print("\n{:*^60}\n", " InternLM2.5 1.5B CLI ");
  fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);
  if (!(prompt_text == "exit" || prompt_text == "quit")) {
    try {
      fmt::print("🔄 Processing...\n");
      mllm::models::internlm2::InternLM2Message prompt{prompt_text};
      auto inputs = tokenizer.convertMessage(prompt);

      fmt::print("\n🤖 Response: ");
      for (auto& step : model.chat(inputs)) {
        auto token = tokenizer.detokenize(step.cur_token_id);
        std::wcout << token << std::flush;
      }
      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }
    model.perfSummary();
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("internlm2_5.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
})
