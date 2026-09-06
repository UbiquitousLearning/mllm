#include <fmt/core.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <mllm/mllm.hpp>
#include <mllm/engine/Context.hpp>
#include <mllm/models/qwen3_5/modeling_qwen3_5.hpp>
#include <mllm/models/qwen3_5/tokenization_qwen3_5.hpp>
#include <mllm/preprocessor/tokenizers/Unicode.hpp>
#include <mllm/utils/AnyValue.hpp>

#include "benchmark_harness.hpp"
#ifdef MLLM_QWEN35_VIDEO_DECODER_BACKEND_PORTABLE
#include "video_decoder.hpp"
#endif

using mllm::Argparse;

MLLM_MAIN({
  auto engine_args = mllm::engineArgAttach();
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer JSON path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Run one prompt non-interactively").required(false);
  auto& prompt_file = Argparse::add<std::string>("--prompt_file").help("Read one benchmark prompt from a file").required(false);
  auto& image_paths = Argparse::add<std::vector<std::string>>("-i|--image_path")
                          .help("Optional still image; repeat in prompt order for multi-image inference")
                          .required(false);
  auto& video_path = Argparse::add<std::string>("--video_path").help("Optional local H.264 MP4 video").required(false);
  auto& video_fps = Argparse::add<float>("--video_fps").help("Target sampled video frames per second").required(false);
  auto& video_max_frames =
      Argparse::add<int>("--video_max_frames").help("Maximum sampled video frames (default 64)").required(false);
  auto& video_max_bytes =
      Argparse::add<int>("--video_max_bytes").help("Maximum MP4 input bytes (default 268435456)").required(false);
  auto& video_max_tokens =
      Argparse::add<int>("--video_max_tokens").help("Maximum projected video tokens (default 4096)").required(false);
  auto& video_max_decoded_pixels = Argparse::add<int64_t>("--video_max_decoded_pixels")
                                       .help("Maximum total decoded source pixels (default 536870912)")
                                       .required(false);
  auto& video_max_selected_pixels = Argparse::add<int64_t>("--video_max_selected_pixels")
                                        .help("Maximum selected source-frame pixels (default 16777216)")
                                        .required(false);
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

  // Argparse validates required options during parse(), so short-circuit help
  // before parsing to make `mllm-qwen3-5-runner --help` usable on its own.
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
      Argparse::printHelp();
      return 0;
    }
  }

  Argparse::parse(argc, argv);
  mllm::configEngineWithArgs(engine_args);

  (void)help;

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  int exit_code = 0;
  {
    mllm::ModelFileVersion file_version;
    if (model_version.get() == "v1") {
      file_version = mllm::ModelFileVersion::kV1;
    } else if (model_version.get() == "v2") {
      file_version = mllm::ModelFileVersion::kV2;
    } else {
      throw std::invalid_argument("model_version must be either v1 or v2");
    }

    auto cfg = mllm::models::qwen3_5::Qwen3_5Config(config_path.get());
    const auto configured_image_paths = image_paths.get();
    const bool has_video = video_path.isSet();
    int generation_limit = max_new_tokens.isSet() ? max_new_tokens.get() : 64;
    if (generation_limit <= 0 || generation_limit > cfg.max_cache_length) {
      throw std::invalid_argument("max_new_tokens must be between 1 and max_cache_length");
    }
    const bool benchmark_mode = benchmark_samples.isSet() || benchmark_jsonl.isSet() || benchmark_variant.isSet()
                                || benchmark_source_sha.isSet() || prompt_file.isSet() || expected_prompt_tokens.isSet()
                                || benchmark_warmup.isSet() || require_device_telemetry.isSet();
    if (prompt.isSet() && prompt_file.isSet()) { throw std::invalid_argument("prompt and prompt_file are mutually exclusive"); }
    if (prompt.isSet() && prompt.get().empty()) { throw std::invalid_argument("prompt must not be empty"); }
    if (image_paths.isSet() && std::any_of(configured_image_paths.begin(), configured_image_paths.end(), [](const auto& path) {
          return path.empty();
        })) {
      throw std::invalid_argument("image_path must not be empty");
    }
    if (image_paths.isSet() && !cfg.vision_enabled) {
      throw std::invalid_argument("image_path requires a config with vision_config");
    }
    if (has_video && video_path.get().empty()) { throw std::invalid_argument("video_path must not be empty"); }
    if (has_video && image_paths.isSet()) {
      throw std::invalid_argument("image_path and video_path are mutually exclusive in the bounded video contract");
    }
    if (has_video && !cfg.vision_enabled) { throw std::invalid_argument("video_path requires a config with vision_config"); }
    if (image_paths.isSet() && benchmark_mode) { throw std::invalid_argument("image_path is not supported in benchmark mode"); }
    if (has_video && benchmark_mode) { throw std::invalid_argument("video_path is not supported in benchmark mode"); }
    const double configured_video_fps = video_fps.isSet() ? video_fps.get() : 2.0;
    constexpr int kVideoFrameCeiling = 64;
    constexpr int kVideoByteCeiling = 268435456;
    constexpr int kVideoTokenCeiling = 4096;
    const int configured_video_max_frames = video_max_frames.isSet() ? video_max_frames.get() : kVideoFrameCeiling;
    const int configured_video_max_bytes = video_max_bytes.isSet() ? video_max_bytes.get() : kVideoByteCeiling;
    const int configured_video_max_tokens = video_max_tokens.isSet() ? video_max_tokens.get() : kVideoTokenCeiling;
    constexpr int64_t kVideoDecodedPixelCeiling = 512LL * 1024 * 1024;
    constexpr int64_t kVideoSelectedPixelCeiling = 16LL * 1024 * 1024;
    const int64_t configured_video_max_decoded_pixels =
        video_max_decoded_pixels.isSet() ? video_max_decoded_pixels.get() : kVideoDecodedPixelCeiling;
    const int64_t configured_video_max_selected_pixels =
        video_max_selected_pixels.isSet() ? video_max_selected_pixels.get() : kVideoSelectedPixelCeiling;
    if (!(configured_video_fps > 0.0) || configured_video_max_frames < 4 || configured_video_max_frames > kVideoFrameCeiling
        || configured_video_max_bytes <= 0 || configured_video_max_bytes > kVideoByteCeiling || configured_video_max_tokens <= 0
        || configured_video_max_tokens > kVideoTokenCeiling || configured_video_max_decoded_pixels <= 0
        || configured_video_max_decoded_pixels > kVideoDecodedPixelCeiling || configured_video_max_selected_pixels <= 0
        || configured_video_max_selected_pixels > kVideoSelectedPixelCeiling) {
      throw std::invalid_argument("video sampling or resource limits are outside the bounded contract");
    }
    if (!has_video
        && (video_fps.isSet() || video_max_frames.isSet() || video_max_bytes.isSet() || video_max_tokens.isSet()
            || video_max_decoded_pixels.isSet() || video_max_selected_pixels.isSet())) {
      throw std::invalid_argument("video sampling and resource limits require video_path");
    }
#ifndef MLLM_QWEN35_VIDEO_DECODER_BACKEND_PORTABLE
    if (has_video) { throw std::invalid_argument("video_path requires MLLM_QWEN35_VIDEO_DECODER_BACKEND=portable"); }
#endif
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
      const std::filesystem::path jsonl_path(benchmark_jsonl.get());
      std::error_code size_error;
      if (std::filesystem::exists(jsonl_path) && std::filesystem::file_size(jsonl_path, size_error) != 0) {
        throw std::invalid_argument("benchmark_jsonl must be new or empty");
      }
    }

#ifdef MLLM_QWEN35_VIDEO_DECODER_BACKEND_PORTABLE
    std::optional<mllm::examples::qwen3_5::DecodedVideo> decoded_video;
    if (has_video) {
      decoded_video = mllm::examples::qwen3_5::decodeH264Mp4Portable(
          video_path.get(), configured_video_fps, 4, configured_video_max_frames,
          static_cast<size_t>(configured_video_max_bytes), configured_video_max_decoded_pixels,
          configured_video_max_selected_pixels);
      const mllm::models::qwen3_5::Qwen3_5VideoPreprocessor video_preprocessor(
          4 * 32 * 32, 24 * 32 * 32 * 1024, cfg.vision_patch_size, cfg.vision_temporal_patch_size,
          cfg.vision_spatial_merge_size);
      const auto [resized_height, resized_width] = video_preprocessor.smartResize(
          decoded_video->source_frame_indices.size(), decoded_video->source_height, decoded_video->source_width);
      const int64_t grid_t =
          (decoded_video->source_frame_indices.size() + cfg.vision_temporal_patch_size - 1) / cfg.vision_temporal_patch_size;
      const int64_t projected_video_tokens = grid_t * (resized_height / cfg.vision_patch_size)
                                             * (resized_width / cfg.vision_patch_size)
                                             / (cfg.vision_spatial_merge_size * cfg.vision_spatial_merge_size);
      if (projected_video_tokens > configured_video_max_tokens) {
        throw std::invalid_argument(fmt::format("projected video tokens ({}) exceed video_max_tokens ({})",
                                                projected_video_tokens, configured_video_max_tokens));
      }
      fmt::print("Video: {} source frames at {:.6g} fps, {}x{}, selected indices=", decoded_video->source_frame_count,
                 decoded_video->source_frames_per_second, decoded_video->source_width, decoded_video->source_height);
      for (size_t index = 0; index < decoded_video->source_frame_indices.size(); ++index) {
        if (index != 0) fmt::print(",");
        fmt::print("{}", decoded_video->source_frame_indices[index]);
      }
      fmt::print("; target_fps={:.6g}; max_frames={}; max_bytes={}; max_decoded_pixels={}; max_selected_pixels={}; "
                 "projected_tokens={}; max_tokens={}\n",
                 configured_video_fps, configured_video_max_frames, configured_video_max_bytes,
                 configured_video_max_decoded_pixels, configured_video_max_selected_pixels, projected_video_tokens,
                 configured_video_max_tokens);
    }
#endif

    auto param = mllm::load(model_path.get(), file_version);
    mllm::models::qwen3_5::validateModelConfigMatch(cfg, param);

    // The chat-template backend comes from config.json (chat_template_backend);
    // a Jinja template is discovered next to tokenizer.json.
    auto tokenizer = mllm::models::qwen3_5::Qwen3_5Tokenizer(tokenizer_path.get(), cfg);
    fmt::print("chat template backend: {}\n", mllm::preprocessor::chatTemplateBackendName(tokenizer.chatTemplateBackend()));
    auto model = mllm::models::qwen3_5::Qwen3_5ForCausalLM(cfg);
    fmt::print("{}: {} layers ({} full attention + {} GDN)\n", mllm::models::qwen3_5::modelNameForConfig(cfg),
               cfg.num_hidden_layers, cfg.numFullAttentionLayers(), cfg.numGDNLayers());

    model.load(param);

    std::string benchmark_prompt;
    if (benchmark_mode) {
      std::ifstream prompt_stream(prompt_file.get(), std::ios::binary);
      if (!prompt_stream) { throw std::invalid_argument("unable to read prompt_file"); }
      benchmark_prompt.assign(std::istreambuf_iterator<char>(prompt_stream), std::istreambuf_iterator<char>());
      while (!benchmark_prompt.empty() && (benchmark_prompt.back() == '\n' || benchmark_prompt.back() == '\r')) {
        benchmark_prompt.pop_back();
      }
      if (benchmark_prompt.empty()) { throw std::invalid_argument("prompt_file must not be empty"); }

      const auto inputs = tokenizer.convertMessage({.prompt = benchmark_prompt});
      const auto prompt_length = inputs.at("sequence").shape()[1];
      if (prompt_length != expected_prompt_tokens.get()) {
        throw std::invalid_argument(
            fmt::format("prompt token count {} differs from expected {}", prompt_length, expected_prompt_tokens.get()));
      }
      if (prompt_length + generation_limit - 1 > cfg.max_cache_length) {
        throw std::invalid_argument("benchmark prompt plus generation exceeds max_cache_length");
      }

      std::ofstream jsonl(benchmark_jsonl.get(), std::ios::out | std::ios::trunc);
      if (!jsonl) { throw std::invalid_argument("unable to open benchmark_jsonl"); }
      const int warmup_count = benchmark_warmup.isSet() ? benchmark_warmup.get() : 0;
      const int total_requests = warmup_count + benchmark_samples.get();
      for (int request_index = 0; request_index < total_requests; ++request_index) {
        const bool warmup = request_index < warmup_count;
        nlohmann::json record = {
            {"schema", "mllm.qwen35.product_benchmark.r4.v1"},
            {"variant", benchmark_variant.get()},
            {"source_sha", benchmark_source_sha.get()},
            {"request_index", request_index},
            {"warmup", warmup},
            {"prompt_file", prompt_file.get()},
            {"prompt_tokens", prompt_length},
            {"max_new_tokens", generation_limit},
            {"cpu_op_threads", mllm::Context::instance().getCpuOpThreads()},
        };
        record["telemetry_before"] = mllm::examples::qwen3_5::benchmark::captureTelemetry();
        std::vector<std::string> invalid_reasons;
        if (require_device_telemetry.isSet() && require_device_telemetry.get()) {
          invalid_reasons = mllm::examples::qwen3_5::benchmark::validateRequiredTelemetry(record["telemetry_before"]);
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
        record["telemetry_after"] = mllm::examples::qwen3_5::benchmark::captureTelemetry();
        if (require_device_telemetry.isSet() && require_device_telemetry.get()) {
          auto after_errors = mllm::examples::qwen3_5::benchmark::validateRequiredTelemetry(record["telemetry_after"]);
          invalid_reasons.insert(invalid_reasons.end(), after_errors.begin(), after_errors.end());
          auto stability_errors = mllm::examples::qwen3_5::benchmark::validateStableTelemetry(record["telemetry_before"],
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
        if (stats.prefill_tokens != prompt_length) { invalid_reasons.push_back("prefill_token_count_mismatch"); }
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
      fmt::print("\n{:*^60}\n", prompt.isSet() ? " Qwen3.5 One-shot CLI " : " Qwen3.5 Interactive CLI ");
      if (!prompt.isSet()) { fmt::print("Enter 'exit' or 'quit' to end the session\n\n"); }

      while (true) {
        std::string prompt_text = prompt.isSet() ? prompt.get() : "";
        if (!prompt.isSet()) {
          fmt::print("Prompt text (or 'exit/quit'): ");
          if (!std::getline(std::cin, prompt_text) || prompt_text == "exit" || prompt_text == "quit") { break; }
        }
        if (prompt_text.empty()) { continue; }

        try {
          // Each prompt is an independent conversation. Both the full-attention
          // KV cache and every GDN recurrent/conv state must start empty.
          model.resetState();
          fmt::print("Processing...\n");
          mllm::models::qwen3_5::Qwen3_5Message message{.prompt = prompt_text, .image_paths = configured_image_paths};
#ifdef MLLM_QWEN35_VIDEO_DECODER_BACKEND_PORTABLE
          if (decoded_video.has_value()) {
            message.video_frames_thwc = decoded_video->frames_thwc;
            message.video_frame_indices = decoded_video->source_frame_indices;
            message.video_frames_per_second = decoded_video->source_frames_per_second;
          }
#endif
          auto inputs = tokenizer.convertMessage(message);
          const auto prompt_length = inputs.at("sequence").shape()[1];
          if (prompt_length + generation_limit - 1 > cfg.max_cache_length) {
            throw std::invalid_argument(fmt::format("prompt token count ({}) plus max_new_tokens ({}) exceeds "
                                                    "max_cache_length ({})",
                                                    prompt_length, generation_limit, cfg.max_cache_length));
          }

          fmt::print("\nResponse: ");
          mllm::models::qwen3_5::Qwen3_5StreamingUtf8Decoder utf8_decoder;

          for (auto& step : model.chat(inputs, {{"max_length", mllm::AnyValue(generation_limit)}})) {
            if (print_token_ids.isSet() && print_token_ids.get()) { fmt::print(stderr, "TOKEN_ID:{}\n", step.cur_token_id); }
            fmt::print("{}", utf8_decoder.append(tokenizer.detokenizeBytes(step.cur_token_id)));
            std::fflush(stdout);
          }
          fmt::print("{}", utf8_decoder.finish());

          fmt::print("\n{}\n", std::string(60, '-'));
        } catch (const std::exception& e) {
          fmt::print("\nError: {}\n{}\n", e.what(), std::string(60, '-'));
          if (prompt.isSet()) { exit_code = 1; }
        }
        if (prompt.isSet()) { break; }
      }

      model.perfSummary();
    }
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3_5.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
  return exit_code;
})
