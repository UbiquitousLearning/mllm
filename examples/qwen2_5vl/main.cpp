#include <mllm/mllm.hpp>
#include <mllm/models/qwen2_5vl/modeling_qwen2_5vl.hpp>
#include <mllm/models/qwen2_5vl/configuration_qwen2_5vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>

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
    auto qwen2vl_cfg = mllm::models::qwen2_5vl::Qwen2_5VLConfig(config_path.get());
    auto qwen2vl_tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path.get());
    // TODO llm

    auto param = mllm::load(model_path.get(), file_version);

    // TODO load

    fmt::print("\n{:*^60}\n", " Qwen2_5VL Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string image_path;
    std::string prompt_text;

    fmt::print("📷 Image path (or 'exit/quit'): ");
    std::getline(std::cin, image_path);

    if (image_path == "exit" || image_path == "quit") { return 0; }

    fmt::print("💬 Prompt text: ");
    std::getline(std::cin, prompt_text);

    try {
      fmt::print("🔄 Processing...\n");
      auto inputs = qwen2vl_tokenizer.convertMessage({.prompt = prompt_text, .img_file_path = image_path});

      auto dict = mllm::models::qwen2_5vl::makeVisualTokensIdBioMap(inputs.grid_thw);
      auto orig_2_win = dict.first;
      auto win_2_orig = dict.second;

      fmt::print("\n🤖 Response: ");

      // TODO

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen2vl.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
})
