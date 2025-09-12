// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>
#include <mllm/models/qwen2_5vl/configuration_qwen2_5vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>

#include "./models/qwen2_5vl/modeling_qwen2_5vl_fast.hpp"
#include "lazy_vlm/models/qwen2_5vl/lazy_vlm_cfg_fast.hpp"

MLLM_MAIN({
  mllm::setLogLevel(mllm::LogLevel::kError);

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

  // 1. prefill_only
  // 2. prefill_decode
  // 3. normal
  auto& settings = Argparse::add<std::string>("-s|--settings").help("Settings path").def("normal");

  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Prompt").required(true);
  auto& image_path = Argparse::add<std::string>("-i|--image_path").help("Image path").required(true);

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
    LazyVLMConfigFast lazy_cfg;

    if (settings.get() == "normal") {
      lazy_cfg.decode_callback = false;
      lazy_cfg.pruning_settings = {};
    } else if (settings.get() == "prefill_only") {
      lazy_cfg.decode_callback = false;
      lazy_cfg.pruning_settings = {
          {3, 0.20}, {6, 0.20}, {9, 0.20}, {12, 0.20}, {15, 0.20}, {18, 0.20},
      };
    } else if (settings.get() == "prefill_decode") {
      lazy_cfg.decode_callback = true;
      lazy_cfg.pruning_settings = {
          {3, 0.20}, {6, 0.20}, {9, 0.20}, {12, 0.20}, {15, 0.20}, {18, 0.20},
      };
    }

    auto qwen2_5vl_cfg = mllm::models::qwen2_5vl::Qwen2_5VLConfig(config_path.get());
    auto qwen2_5vl_tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path.get(), 56 * 56, 28 * 28 * 2048);
    auto qwen2_5vl = Qwen2_5VLForCausalLM(qwen2_5vl_cfg, lazy_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    qwen2_5vl.llm.load(param);
    qwen2_5vl.visual.load(param);

    try {
      fmt::print("🔄 Processing...\n");
      auto inputs = qwen2_5vl_tokenizer.convertMessage({.prompt = prompt.get(), .img_file_path = image_path.get()});

      fmt::print("\n🤖 Response: ");

      for (auto& step : qwen2_5vl.chat(inputs, {
                                                   {"do_sample", mllm::AnyValue(false)},
                                                   {"max_length", mllm::AnyValue(qwen2_5vl_cfg.max_cache_length)},
                                               })) {
        std::wcout << qwen2_5vl_tokenizer.detokenize(step.cur_token_id) << std::flush;
      }

      qwen2_5vl.perfSummary();

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen2vl.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
});
