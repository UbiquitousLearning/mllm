// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <mllm/mllm.hpp>
#include <mllm/engine/Context.hpp>
#include <mllm/models/minicpm5/modeling_minicpm5.hpp>
#include <mllm/models/minicpm5/tokenization_minicpm5.hpp>
#include <mllm/utils/AnyValue.hpp>

#include "benchmark_harness.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto engine_args = mllm::engineArgAttach();
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Converted model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version: v1 or v2").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Official tokenizer.json path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Runtime config path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Run one prompt non-interactively").required(false);
  auto& prompt_file = Argparse::add<std::string>("--prompt_file").help("Read one benchmark prompt from a file").required(false);
  auto& system = Argparse::add<std::string>("--system").help("Optional system message").required(false);
  auto& enable_thinking =
      Argparse::add<bool>("--enable_thinking").help("Open a thinking block in the generation prompt").required(false);
  auto& max_new_tokens = Argparse::add<int>("-g|--max_new_tokens").help("Maximum generated tokens per prompt").required(false);
  auto& print_token_ids = Argparse::add<bool>("--print_token_ids").help("Print generated token IDs to stderr").required(false);
  auto& benchmark_warmup =
      Argparse::add<int>("--benchmark_warmup").help("Unrecorded benchmark warmup requests").required(false);
  auto& benchmark_samples = Argparse::add<int>("--benchmark_samples").help("Measured benchmark requests").required(false);
  auto& benchmark_jsonl = Argparse::add<std::string>("--benchmark_jsonl").help("Fresh JSONL output path").required(false);
  auto& benchmark_variant = Argparse::add<std::string>("--benchmark_variant").help("Bound variant identity").required(false);
  auto& benchmark_source_sha = Argparse::add<std::string>("--benchmark_source_sha").help("Bound source SHA").required(false);
  auto& expected_prompt_tokens =
      Argparse::add<int>("--expected_prompt_tokens").help("Fail if tokenized prompt length differs").required(false);
  auto& require_device_telemetry = Argparse::add<bool>("--require_device_telemetry")
                                       .help("Fail closed on incomplete or drifting CPU telemetry")
                                       .required(false);

  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) == "-h" || std::string(argv[index]) == "--help") {
      Argparse::printHelp();
      return 0;
    }
  }
  Argparse::parse(argc, argv);
  mllm::configEngineWithArgs(engine_args);
  (void)help;

  mllm::ModelFileVersion file_version;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  } else {
    throw std::invalid_argument("model_version must be either v1 or v2");
  }

  int exit_code = 0;
  {
    const auto config = mllm::models::minicpm5::MiniCPM5Config(config_path.get());
    int generation_limit = max_new_tokens.isSet() ? max_new_tokens.get() : 64;
    if (generation_limit <= 0 || generation_limit > config.max_cache_length) {
      throw std::invalid_argument("max_new_tokens must be between 1 and max_cache_length");
    }
    const bool benchmark_mode = benchmark_samples.isSet() || benchmark_jsonl.isSet() || benchmark_variant.isSet()
                                || benchmark_source_sha.isSet() || prompt_file.isSet() || expected_prompt_tokens.isSet()
                                || benchmark_warmup.isSet() || require_device_telemetry.isSet();
    if (prompt.isSet() && prompt_file.isSet()) { throw std::invalid_argument("prompt and prompt_file are mutually exclusive"); }
    if (prompt.isSet() && prompt.get().empty()) { throw std::invalid_argument("prompt must not be empty"); }
    if (benchmark_mode) {
      if (!prompt_file.isSet() || !benchmark_samples.isSet() || !benchmark_jsonl.isSet() || !benchmark_variant.isSet()
          || !benchmark_source_sha.isSet() || !expected_prompt_tokens.isSet()) {
        throw std::invalid_argument("benchmark mode requires prompt_file, benchmark_samples, benchmark_jsonl, "
                                    "benchmark_variant, benchmark_source_sha, and expected_prompt_tokens");
      }
      if (benchmark_samples.get() <= 0) { throw std::invalid_argument("benchmark_samples must be positive"); }
      if (benchmark_warmup.isSet() && benchmark_warmup.get() < 0) {
        throw std::invalid_argument("benchmark_warmup must be non-negative");
      }
      if (generation_limit < 2) { throw std::invalid_argument("benchmark max_new_tokens must be at least 2"); }
      if (benchmark_variant.get().empty() || benchmark_source_sha.get().empty()) {
        throw std::invalid_argument("benchmark identities must not be empty");
      }
      if (system.isSet()) { throw std::invalid_argument("benchmark mode does not accept a system message"); }
      if (enable_thinking.isSet() && enable_thinking.get()) {
        throw std::invalid_argument("benchmark mode requires enable_thinking=false");
      }
      const std::filesystem::path jsonl_path(benchmark_jsonl.get());
      std::error_code size_error;
      if (std::filesystem::exists(jsonl_path) && std::filesystem::file_size(jsonl_path, size_error) != 0) {
        throw std::invalid_argument("benchmark_jsonl must be new or empty");
      }
    }

    auto parameters = mllm::load(model_path.get(), file_version);
    mllm::models::minicpm5::validateModelConfigMatch(config, parameters);
    // The chat-template backend comes from config.json (chat_template_backend);
    // a Jinja template is discovered next to tokenizer.json.
    auto tokenizer = mllm::models::minicpm5::MiniCPM5Tokenizer(tokenizer_path.get(), config);
    fmt::print("chat template backend: {}\n", mllm::preprocessor::chatTemplateBackendName(tokenizer.chatTemplateBackend()));
    auto model = mllm::models::minicpm5::MiniCPM5ForCausalLM(config);
    model.load(parameters);

    if (benchmark_mode) {
      std::ifstream prompt_stream(prompt_file.get(), std::ios::binary);
      if (!prompt_stream) { throw std::invalid_argument("unable to read prompt_file"); }
      std::string benchmark_prompt;
      benchmark_prompt.assign(std::istreambuf_iterator<char>(prompt_stream), std::istreambuf_iterator<char>());
      while (!benchmark_prompt.empty() && (benchmark_prompt.back() == '\n' || benchmark_prompt.back() == '\r')) {
        benchmark_prompt.pop_back();
      }
      if (benchmark_prompt.empty()) { throw std::invalid_argument("prompt_file must not be empty"); }

      const auto inputs = tokenizer.convertMessage({.prompt = benchmark_prompt});
      const auto sequence = inputs.at("sequence");
      const int32_t prompt_tokens = sequence.shape()[1];
      if (prompt_tokens != expected_prompt_tokens.get()) {
        throw std::invalid_argument(
            fmt::format("prompt token count {} differs from expected {}", prompt_tokens, expected_prompt_tokens.get()));
      }
      if (prompt_tokens + generation_limit - 1 > config.max_cache_length) {
        throw std::invalid_argument("benchmark prompt plus generation exceeds max_cache_length");
      }
      const std::vector<int64_t> prompt_token_ids(sequence.ptr<int64_t>(), sequence.ptr<int64_t>() + sequence.numel());

      std::ofstream jsonl(benchmark_jsonl.get(), std::ios::out | std::ios::trunc);
      if (!jsonl) { throw std::invalid_argument("unable to open benchmark_jsonl"); }
      const int warmup_count = benchmark_warmup.isSet() ? benchmark_warmup.get() : 0;
      const int total_requests = warmup_count + benchmark_samples.get();
      for (int request_index = 0; request_index < total_requests; ++request_index) {
        const bool warmup = request_index < warmup_count;
        nlohmann::json record = {
            {"schema", "mllm.minicpm5.product_benchmark.v1"},
            {"variant", benchmark_variant.get()},
            {"source_sha", benchmark_source_sha.get()},
            {"request_index", request_index},
            {"warmup", warmup},
            {"prompt_file", prompt_file.get()},
            {"prompt_tokens", prompt_tokens},
            {"prompt_token_ids", prompt_token_ids},
            {"max_new_tokens", generation_limit},
            {"min_new_tokens", generation_limit},
            {"do_sample", false},
            {"enable_thinking", false},
            {"cpu_op_threads", mllm::Context::instance().getCpuOpThreads()},
        };
        record["telemetry_before"] = mllm::examples::minicpm5::benchmark::captureTelemetry();
        std::vector<std::string> invalid_reasons;
        if (require_device_telemetry.isSet() && require_device_telemetry.get()) {
          invalid_reasons = mllm::examples::minicpm5::benchmark::validateRequiredTelemetry(record["telemetry_before"]);
        }

        const auto reset_start = std::chrono::steady_clock::now();
        model.resetState();
        const auto reset_end = std::chrono::steady_clock::now();
        std::vector<int64_t> generated_token_ids;
        const auto request_start = std::chrono::steady_clock::now();
        model.streamGenerate(inputs,
                             {{"max_length", mllm::AnyValue(generation_limit)},
                              {"min_new_tokens", mllm::AnyValue(generation_limit)},
                              {"do_sample", mllm::AnyValue(false)}},
                             [&](int64_t token_id) { generated_token_ids.push_back(token_id); });
        const auto request_end = std::chrono::steady_clock::now();
        record["telemetry_after"] = mllm::examples::minicpm5::benchmark::captureTelemetry();
        if (require_device_telemetry.isSet() && require_device_telemetry.get()) {
          auto after_errors = mllm::examples::minicpm5::benchmark::validateRequiredTelemetry(record["telemetry_after"]);
          invalid_reasons.insert(invalid_reasons.end(), after_errors.begin(), after_errors.end());
          auto stability_errors = mllm::examples::minicpm5::benchmark::validateStableTelemetry(record["telemetry_before"],
                                                                                               record["telemetry_after"]);
          invalid_reasons.insert(invalid_reasons.end(), stability_errors.begin(), stability_errors.end());
        }

        const auto stats = model.perfStats();
        record["generated_token_ids"] = generated_token_ids;
        record["reset_duration_us"] = std::chrono::duration_cast<std::chrono::microseconds>(reset_end - reset_start).count();
        record["request_wall_duration_us"] =
            std::chrono::duration_cast<std::chrono::microseconds>(request_end - request_start).count();
        record["stats"] = {
            {"valid", stats.valid},
            {"completed", stats.completed},
            {"total_duration_us", stats.total_duration_us},
            {"prefill_duration_us", stats.prefill_duration_us},
            {"decode_duration_us", stats.decode_duration_us},
            {"ttft_duration_us", stats.ttft_duration_us},
            {"prefill_tokens", stats.prefill_tokens},
            {"generated_tokens", stats.generated_tokens},
            {"decode_steps", stats.decode_steps},
        };
        if (!stats.valid) { invalid_reasons.push_back("invalid_performance_stats"); }
        if (!stats.completed) { invalid_reasons.push_back("incomplete_generation"); }
        if (stats.prefill_tokens != prompt_tokens) { invalid_reasons.push_back("prefill_token_count_mismatch"); }
        if (stats.generated_tokens != generation_limit || stats.decode_steps != generation_limit - 1
            || generated_token_ids.size() != static_cast<size_t>(generation_limit)) {
          invalid_reasons.push_back("generation_length_mismatch");
        }
        record["invalid_reasons"] = invalid_reasons;
        record["status"] = invalid_reasons.empty() ? "ok" : "invalid";
        jsonl << record.dump() << '\n';
        jsonl.flush();
        if (!jsonl) { throw std::runtime_error("failed to write benchmark_jsonl"); }
        if (!invalid_reasons.empty()) {
          exit_code = 2;
          break;
        }
      }
      if (exit_code == 0) { fmt::print("Benchmark records: {}\n", benchmark_jsonl.get()); }
    } else {
      fmt::print("\n{:*^60}\n", prompt.isSet() ? " MiniCPM5-1B One-shot CLI " : " MiniCPM5-1B Interactive CLI ");
      if (!prompt.isSet()) fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

      while (true) {
        std::string prompt_text = prompt.isSet() ? prompt.get() : "";
        if (!prompt.isSet()) {
          fmt::print("Prompt text (or 'exit/quit'): ");
          if (!std::getline(std::cin, prompt_text) || prompt_text == "exit" || prompt_text == "quit") break;
        }
        if (prompt_text.empty()) continue;

        try {
          model.resetState();
          const auto inputs = tokenizer.convertMessage({
              .prompt = prompt_text,
              .system = system.isSet() ? system.get() : "",
              .enable_thinking = enable_thinking.isSet() && enable_thinking.get(),
          });
          const int32_t prompt_tokens = inputs.at("sequence").shape()[1];
          if (prompt_tokens + generation_limit - 1 > config.max_cache_length) {
            throw std::invalid_argument(fmt::format("prompt token count ({}) plus max_new_tokens ({}) exceeds "
                                                    "max_cache_length ({})",
                                                    prompt_tokens, generation_limit, config.max_cache_length));
          }

          fmt::print("\nResponse: ");
          mllm::models::minicpm5::MiniCPM5StreamingUtf8Decoder decoder;
          for (auto& step : model.chat(inputs, {{"max_length", mllm::AnyValue(generation_limit)}})) {
            if (print_token_ids.isSet() && print_token_ids.get()) fmt::print(stderr, "TOKEN_ID:{}\n", step.cur_token_id);
            fmt::print("{}", decoder.append(tokenizer.detokenizeBytes(step.cur_token_id)));
            std::fflush(stdout);
          }
          fmt::print("{}\n{}\n", decoder.finish(), std::string(60, '-'));
        } catch (const std::exception& error) {
          fmt::print("\nError: {}\n{}\n", error.what(), std::string(60, '-'));
          if (prompt.isSet()) exit_code = 1;
        }
        if (prompt.isSet()) break;
      }
      model.perfSummary();
    }
  }

  mllm::print("\n");
  mllm::memoryReport();
  return exit_code;
})
