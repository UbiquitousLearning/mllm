#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>

using mllm::Argparse;

void generate(mllm::models::qwen2vl::Qwen2VLForCausalLM& model, mllm::models::qwen2vl::Qwen2VLConfig& cfg,
              mllm::models::qwen2vl::Qwen2VLTokenizer& tokenizer, mllm::models::qwen2vl::Qwen2VLForCausalLMOutputPast& input) {
  auto o = model(input);

  int64_t pos_idx = -1;

  int32_t l = 0;

  // Greedy decoding one token
  while (true) {
    l++;
    // [B, S, D]
    auto sequence = o.sequence;
    auto S = sequence.shape()[1];
    auto D = sequence.shape()[2];
    auto sequence_ptr = sequence.offsettedPtr<float>({0, S - 1, 0});
    auto max_logits_idx_ptr = std::max_element(sequence_ptr, sequence_ptr + D);
    pos_idx = std::distance(sequence_ptr, max_logits_idx_ptr);
    auto str = tokenizer.detokenize(pos_idx);

    if (!(pos_idx != cfg.eos_token_id && pos_idx != cfg.end_of_text_token_id && l < cfg.max_cache_length)) { break; }

    std::wcout << str << std::flush;

    // Generate new input
    sequence = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
    sequence.ptr<int64_t>()[0] = pos_idx;
    o.sequence = sequence;
    o = model(o);
  }
}

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

  Argparse::parse(argc, argv);

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
    auto qwen2vl_cfg = mllm::models::qwen2vl::Qwen2VLConfig(config_path.get());
    auto qwen2vl_tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path.get());
    auto qwen2vl = mllm::models::qwen2vl::Qwen2VLForCausalLM(qwen2vl_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    qwen2vl.llm.load(param);
    qwen2vl.visual.load(param);

    fmt::print("\n{:*^60}\n", " Qwen2VL Interactive CLI ");
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

      fmt::print("\n🤖 Response: ");
      generate(qwen2vl, qwen2vl_cfg, qwen2vl_tokenizer, inputs);
      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }
  }

  mllm::print("\n");
  mllm::memoryReport();
})
