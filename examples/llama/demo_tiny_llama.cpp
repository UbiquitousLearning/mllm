#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include "mllm/models/llama/modeling_llama.hpp"
#include "mllm/models/llama/tokenization_tiny_llama.hpp"
#include "mllm/models/llama/configuration_llama.hpp"
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
    auto llama_cfg = mllm::models::llama::LLaMAConfig(config_path.get());
    auto llama_tokenizer = mllm::models::llama::TinyLlamaTokenizer(tokenizer_path.get());
    auto llama = mllm::models::llama::LlamaForCausalLM("", llama_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    llama.load(param);

    fmt::print("\n{:*^60}\n", " LLaMA Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text;

    while (true) {
      fmt::print("💬 Prompt text (or 'exit/quit'): ");
      std::getline(std::cin, prompt_text);

      if (prompt_text == "exit" || prompt_text == "quit") { break; }

      if (prompt_text.empty()) { continue; }

      try {
        fmt::print("🔄 Processing...\n");
        auto inputs = llama_tokenizer.convertMessage({
            {
                .role = "system",
                .content = "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {.role = "user", .content = prompt_text},
        });

        fmt::print("\n🤖 Response: ");

        llama.streamGenerate(inputs,
                             {
                                 {"do_sample", mllm::AnyValue(false)},
                                 {"max_length", mllm::AnyValue(10)},
                             },
                             [&](int64_t token_id) {
                               auto str = llama_tokenizer.detokenize(token_id);
                               std::wcout << str << std::flush;
                             });

        fmt::print("\n{}\n", std::string(60, '-'));
      } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }
      break;
    }
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("llama.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();

  return 0;
})