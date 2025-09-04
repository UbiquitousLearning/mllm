// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>
#include <mllm/models/qwen2_5vl/configuration_qwen2_5vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>

#include "./models/qwen2_5vl/modeling_qwen2_5vl.hpp"
#include "lazy_vlm/models/qwen2_5vl/lazy_vlm_cfg.hpp"

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
    auto qwen2_5vl_cfg = mllm::models::qwen2_5vl::Qwen2_5VLConfig(config_path.get());
    auto qwen2_5vl_tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path.get());
    auto qwen2_5vl = Qwen2_5VLForCausalLM(qwen2_5vl_cfg, LazyVLMConfig{.decode_callback = false,
                                                                       .pruning_settings = {
                                                                           //  {3, 0.0},
                                                                           //  {6, 0.0},
                                                                           //  {9, 0.0},
                                                                           //  {12, 0.0},
                                                                           //  {15, 0.0},
                                                                           //  {18, 0.0},
                                                                       }});

    auto param = mllm::load(model_path.get(), file_version);
    qwen2_5vl.llm.load(param);
    qwen2_5vl.visual.load(param);

    try {
      fmt::print("🔄 Processing...\n");
      auto inputs = qwen2_5vl_tokenizer.convertMessage(
          {.prompt = "Describe this image", .img_file_path = "/Volumes/D/mllm/.tmp/gafei.jpeg"});

      fmt::print("\n🤖 Response: ");

      // Steam it!
      qwen2_5vl.streamGenerate(inputs,
                               {
                                   {"do_sample", mllm::AnyValue(false)},
                                   {"max_length", mllm::AnyValue(qwen2_5vl_cfg.max_cache_length)},
                               },
                               [&](int64_t token_id) {
                                 auto str = qwen2_5vl_tokenizer.detokenize(token_id);
                                 std::wcout << str << std::flush;
                               });

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
