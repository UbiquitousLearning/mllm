// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <iostream>
#include <clocale>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen_ascend/modeling_qwen_ascend.hpp>
#include <mllm/models/qwen_ascend/tokenization_qwen_ascend.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(false);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& seq_len = Argparse::add<int>("-s|--seq_len").help("Input sequence length for test").required(false);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer json path for QA test").required(false);
  auto& prompt = Argparse::add<std::string>("-p|--prompt")
                     .help("Question prompt for QA generation test")
                     .required(false);
  auto& max_new_tokens =
      Argparse::add<int>("-g|--max_new_tokens").help("Max new tokens for QA generation").required(false);

  Argparse::parse(argc, argv);

  // Keep runtime output concise: hide verbose [INFO] backend allocator logs.
  mllm::Logger::level() = mllm::LogLevel::kWarn;

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.isSet()) {
    if (model_version.get() == "v1") {
      file_version = mllm::ModelFileVersion::kV1;
    } else if (model_version.get() == "v2") {
      file_version = mllm::ModelFileVersion::kV2;
    }
  }

  int test_seq_len = seq_len.isSet() ? seq_len.get() : 8;
  int gen_max_new_tokens = max_new_tokens.isSet() ? max_new_tokens.get() : 64;
  std::string prompt_text = prompt.isSet() ? prompt.get() : "请用一句话介绍你自己。";

  fmt::print("\n{:*^60}\n", " Qwen Ascend Backend Test ");
  fmt::print("Config: {}\n", config_path.get());
  fmt::print("Model: {}\n", model_path.get());
  fmt::print("Test sequence length: {}\n\n", test_seq_len);
  std::setlocale(LC_ALL, "");

#ifdef MLLM_BUILD_ASCEND_BACKEND
  // Register Ascend backend using the unified initialization function
  fmt::print("Registering Ascend backend...\n");
  mllm::initAscendBackend();
  fmt::print("Ascend backend registered successfully.\n\n");
#else
  fmt::print("Warning: Ascend backend not enabled, using CPU backend.\n\n");
#endif

  {
    // Load config
    fmt::print("Loading config...\n");
    auto cfg = mllm::models::qwen_ascend::QwenAscendConfig(config_path.get());
    fmt::print("Config loaded: hidden_size={}, num_layers={}, num_heads={}\n",
               cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads);
    fmt::print("cfg.tie_word_embeddings: {}\n", cfg.tie_word_embeddings ? "true" : "false");

    // Create model
    fmt::print("\nCreating model...\n");
    auto model = mllm::models::qwen_ascend::QwenAscendForCausalLM(cfg);
    fmt::print("Model created.\n");

    // Load weights
    fmt::print("\nLoading weights from {}...\n", model_path.get());
    auto param = mllm::load(model_path.get(), file_version);
    model.load(param);
    fmt::print("Weights loaded.\n");

    // DEBUG: Verify embedding weights on CPU before moving to Ascend
    {
      auto embed_weight = model.debugEmbeddingWeight();
      fmt::print("[DEBUG] Embedding weight (CPU): shape=[{},{}], dtype={}, device={}\n",
                 embed_weight.shape()[0], embed_weight.shape()[1],
                 static_cast<int>(embed_weight.dtype()), static_cast<int>(embed_weight.device()));

      if (embed_weight.device() == mllm::kCPU) {
        float min_v = 0, max_v = 0, sum_v = 0;
        int64_t numel = embed_weight.numel();
        int64_t count = numel > 10000 ? 10000 : numel;
        if (embed_weight.dtype() == mllm::kFloat32) {
          auto* ptr = embed_weight.ptr<float>();
          min_v = ptr[0]; max_v = ptr[0];
          for (int64_t i = 0; i < count; ++i) {
            sum_v += ptr[i];
            if (ptr[i] < min_v) min_v = ptr[i];
            if (ptr[i] > max_v) max_v = ptr[i];
          }
        } else if (embed_weight.dtype() == mllm::kFloat16) {
          auto* ptr = embed_weight.ptr<mllm::mllm_fp16_t>();
          min_v = static_cast<float>(ptr[0]); max_v = min_v;
          for (int64_t i = 0; i < count; ++i) {
            float v = static_cast<float>(ptr[i]);
            sum_v += v;
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
          }
        }
        fmt::print("[DEBUG] Embedding weight stats (first {}): min={:.6f}, max={:.6f}, mean={:.6f}\n",
                   count, min_v, max_v, sum_v / count);
      }
    }

#ifdef MLLM_BUILD_ASCEND_BACKEND
    // Move model to Ascend backend AFTER loading weights so layer ops are recreated on Ascend.
    fmt::print("\nMoving model to Ascend backend...\n");
    model.to(mllm::kAscend);
    fmt::print("Model moved to Ascend.\n");
#endif

    // Create test input (simple token sequence)
    fmt::print("\nCreating test input (seq_len={})...\n", test_seq_len);
    auto input_ids = mllm::Tensor::empty({1, test_seq_len}, mllm::kInt64, mllm::kCPU).alloc();
    auto input_ptr = input_ids.ptr<int64_t>();
    // Fill with some token IDs (e.g., 1, 2, 3, ...)
    for (int i = 0; i < test_seq_len; ++i) {
      input_ptr[i] = i + 1;
    }
    fmt::print("Input created: shape=[1, {}]\n", test_seq_len);

    // Clear KV cache before tests
    model.clearCache();

    // Run forward pass
    fmt::print("\n{:*^60}\n", " Running Forward Pass ");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto logits = model.forward(input_ids);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    fmt::print("Forward pass completed in {} ms\n", duration.count());
    fmt::print("Output logits shape: [{}, {}, {}]\n",
               logits.shape()[0], logits.shape()[1], logits.shape()[2]);
    fmt::print("logits dtype/device(before to CPU): {}/{}\n",
               static_cast<int>(logits.dtype()), mllm::deviceTypes2Str(logits.device()));

    // Move logits to CPU if on Ascend
    if (logits.device() != mllm::kCPU) {
      logits = logits.to(mllm::kCPU);
    }
    fmt::print("logits dtype/device(after to CPU): {}/{}\n",
               static_cast<int>(logits.dtype()), mllm::deviceTypes2Str(logits.device()));

    // Print some output statistics
    int vocab_size = cfg.vocab_size;
    float min_val = 0, max_val = 0, sum = 0;
    int64_t argmax_idx = 0;

    // Handle FP16 data type
    if (logits.dtype() == mllm::kFloat16) {
      auto logits_ptr = logits.ptr<mllm::mllm_fp16_t>();
      min_val = static_cast<float>(logits_ptr[0]);
      max_val = min_val;
      for (int i = 0; i < vocab_size; ++i) {
        float val = static_cast<float>(logits_ptr[i]);
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) {
          max_val = val;
          argmax_idx = i;
        }
      }
    } else {
      auto logits_ptr = logits.ptr<float>();
      min_val = logits_ptr[0];
      max_val = min_val;
      for (int i = 0; i < vocab_size; ++i) {
        float val = logits_ptr[i];
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) {
          max_val = val;
          argmax_idx = i;
        }
      }
    }
    float mean = sum / vocab_size;

    fmt::print("\nLogits statistics:\n");
    fmt::print("  Min: {:.4f}\n", min_val);
    fmt::print("  Max: {:.4f}\n", max_val);
    fmt::print("  Mean: {:.4f}\n", mean);
    fmt::print("  Argmax token ID: {}\n", argmax_idx);

    // Optional QA test with tokenizer + streaming generation.
    if (tokenizer_path.isSet()) {
      fmt::print("\n{:*^60}\n", " QA Generation Test ");
      fmt::print("Tokenizer: {}\n", tokenizer_path.get());
      fmt::print("Prompt: {}\n", prompt_text);
      fmt::print("Max new tokens: {}\n", gen_max_new_tokens);

      auto tokenizer = mllm::models::qwen_ascend::QwenAscendTokenizer(tokenizer_path.get());
      mllm::models::qwen_ascend::QwenAscendMessage msg;
      msg.prompt = prompt_text;
      auto inputs = tokenizer.convertMessage(msg);

      // Clear KV cache before generation
      model.clearCache();

      fmt::print("\nAnswer:\n");
      auto chat_start = std::chrono::high_resolution_clock::now();

      std::vector<int64_t> generated_ids;
      // Use streaming generation with the ARGeneration chat interface
      for (auto& step : model.chat(inputs)) {
        generated_ids.push_back(step.cur_token_id);
        std::wcout << tokenizer.detokenize(step.cur_token_id) << std::flush;
        // Stop if we've reached max_new_tokens
        if (static_cast<int>(generated_ids.size()) >= gen_max_new_tokens) {
          break;
        }
      }
      std::wcout << std::endl;

      auto chat_end = std::chrono::high_resolution_clock::now();
      auto chat_ms = std::chrono::duration_cast<std::chrono::milliseconds>(chat_end - chat_start).count();

      fmt::print("\nGenerated {} tokens in {} ms\n", generated_ids.size(), chat_ms);

      // Print performance summary
      model.perfSummary();
    } else {
      fmt::print("\nTip: Add -t tokenizer.json and -p \"你的问题\" to run a real QA generation test.\n");
    }

    fmt::print("\n{:*^60}\n", " Test Completed ");
  }

  mllm::memoryReport();
  mllm::shutdownContext();
  return 0;
})
