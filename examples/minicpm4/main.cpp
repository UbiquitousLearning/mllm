#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/minicpm4/modeling_minicpm4.hpp>
#include <mllm/models/minicpm4/tokenization_minicpm4.hpp>
#include <mllm/utils/AnyValue.hpp>
#include <mllm/models/ARGeneration.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  /*
    FOR RUN(MacOS Apple Silicon):
      python task.py tasks/build_osx_apple_silicon.yaml
      cd build-osx/bin
      ./mllm-minicpm4-runner -m ../../models/minicpm4.mllm -mv v1 -t ../../tokenizer/MiniCPM4/tokenizer.json -c
    ../../examples/minicpm4/config_minicpm4.json (need to get model.mllm and tokenizer.json first)
  */

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
    auto minicpm4_cfg = mllm::models::minicpm4::MiniCPM4Config(config_path.get());
    auto minicpm4_tokenizer = mllm::models::minicpm4::MiniCPM4Tokenizer(tokenizer_path.get());
    auto minicpm4 = mllm::models::minicpm4::MiniCPM4ForCausalLM(minicpm4_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    minicpm4.load(param);

    fmt::print("\n{:*^60}\n", " MiniCPM4 Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text = "你是谁？";

    try {
      auto inputs = minicpm4_tokenizer.convertMessage({.prompt = prompt_text});

      fmt::print("\nResponse: ");

      mllm::models::ARGenerationArgs args;
      args["temperature"] = mllm::AnyValue(0.8f);
      args["top_p"] = mllm::AnyValue(0.9f);
      args["do_sample"] = mllm::AnyValue(true);

      for (auto& step : minicpm4.chat(inputs, args)) {
        auto token_str = minicpm4_tokenizer.detokenize(step.cur_token_id);
        if (token_str == "<|im_end|>") { break; }
        std::cout << token_str << std::flush;
      }

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\nError: {}\n{}\n", e.what(), std::string(60, '-')); }

    minicpm4.perfSummary();
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("minicpm4.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
})
