// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>

#include <mllm/mllm.hpp>
#include <mllm/utils/Argparse.hpp>
#include <mllm/utils/CPUArchHelper.hpp>
#include <mllm/utils/PlatformRTHelper.hpp>

#define STRINGIFY_INTERNAL(x) #x
#define STRINGIFY(x) STRINGIFY_INTERNAL(x)

#include "models/All.hpp"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

MLLM_MAIN({
  auto& help = mllm::Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_name = mllm::Argparse::add<std::string>("-n|--model_name").help("Model name");
  auto& model_path = mllm::Argparse::add<std::string>("-m|--model_path").help("Model path");
  auto& config_path = mllm::Argparse::add<std::string>("-c|--config_path").help("Config path");
  auto& num_threads = mllm::Argparse::add<int32_t>("-t|--threads").help("Number of threads");
  auto& pp = mllm::Argparse::add<std::string>("-pp|--prompt_length").help("Prompt length");
  auto& tg = mllm::Argparse::add<std::string>("-tg|--test_generation_length").help("Test Generation length");
  auto& cache_length = mllm::Argparse::add<int32_t>("-cl|--cache_length").help("Cache length");

  auto& runs = mllm::Argparse::add<int32_t>("-r|--runs").help("Number of benchmark runs").def(3);
  auto& cooldown_s = mllm::Argparse::add<int32_t>("-cs|--cooldown_s").help("Cooldown time between runs in seconds").def(5);
  auto& output_csv = mllm::Argparse::add<std::string>("-oc|--output_csv").help("Output results to a CSV file").def("");
  auto& schema_version = mllm::Argparse::add<int32_t>("-sv|--schema_version").help("Schema version for output format").def(1);
  auto& kv_dtype_bytes =
      mllm::Argparse::add<int32_t>("-kv|--kv_dtype_bytes").help("KV cache data type bytes (1: int8, 2: fp16, 4: fp32)").def(4);

  mllm::Argparse::parse(argc, argv);

  mllm::Context::instance().setCpuOpThreads(num_threads.get());
  mllm::setMaximumNumThreads((uint32_t)num_threads.get());

  mllm::print("MLLM Build Version :", STRINGIFY(MLLM_GIT_COMMIT_HASH));

  mllm::print("ARCH               :", mllm::cpu::CURRENT_ARCH_STRING);
  mllm::print("FP16               :", mllm::cpu::hasFP16());
  mllm::print("BF16               :", mllm::cpu::hasBF16());
  mllm::print("SVE                :", mllm::cpu::hasSVE());
  mllm::print("SME                :", mllm::cpu::hasSME());
  mllm::print("Neon               :", mllm::cpu::hasNEON());
  mllm::print("DotProd            :", mllm::cpu::hasDotProd());
  mllm::print("SSE                :", mllm::cpu::hasSSE());
  mllm::print("SSE2               :", mllm::cpu::hasSSE2());
  mllm::print("SSE3               :", mllm::cpu::hasSSE3());
  mllm::print("SSSE3              :", mllm::cpu::hasSSSE3());
  mllm::print("SSE4_1             :", mllm::cpu::hasSSE4_1());
  mllm::print("SSE4_2             :", mllm::cpu::hasSSE4_2());
  mllm::print("AVX                :", mllm::cpu::hasAVX());
  mllm::print("AVX2               :", mllm::cpu::hasAVX2());
  mllm::print("AVX512F            :", mllm::cpu::hasAVX512F());
  mllm::print("AVX512BW           :", mllm::cpu::hasAVX512BW());
  mllm::print("AVX512CD           :", mllm::cpu::hasAVX512CD());
  mllm::print("AVX512DQ           :", mllm::cpu::hasAVX512DQ());
  mllm::print("AVX512VL           :", mllm::cpu::hasAVX512VL());
  mllm::print("FMA                :", mllm::cpu::hasFMA());

  mllm::print("Create Benchmark: ", model_name.get());
  auto benchmark = createBenchmark(model_name.get());
  MLLM_RT_ASSERT(benchmark != nullptr);

  int R = runs.get();
  if (R <= 0) {
    mllm::print("[ERROR] --runs must be > 0, got:", R);
    return 1;
  }

  std::ofstream csv_file;
  if (!output_csv.get().empty()) {
    csv_file.open(output_csv.get());
    if (!csv_file.is_open()) {
      mllm::print("[ERROR] Failed to open --output_csv:", output_csv.get());
      return 1;
    }
    csv_file << "schema_version,git_commit,arch,model_name,cache_length,pp,tg,ttft_ms,prefill_speed,decode_speed,prefill_ms,decode_ms_per_"
                "tok,kv_est_bytes_pp,kv_est_bytes_final\n";
  }

  mllm::print("Model Info");
  benchmark->init(config_path.get(), model_path.get(), cache_length.get());
  benchmark->printModelInfo();
  mllm::print("Cache Length       :", cache_length.get());

  // Warmup run
  mllm::print("Warmup Run");
  benchmark->warmup();

  // Split pp and tg if they have multiple set.
  std::vector<std::pair<int32_t, int32_t>> pp_tg_pairs;
  {
    // pp and tg are strings with multiple values separated by comma. We need to split them.
    std::vector<int32_t> pp_values;
    std::vector<int32_t> tg_values;

    // Split pp string
    std::string pp_str = pp.get();
    std::string item;
    std::istringstream pp_stream(pp_str);
    while (std::getline(pp_stream, item, ',')) { pp_values.push_back(std::stoi(item)); }

    // Split tg string
    std::string tg_str = tg.get();
    std::istringstream tg_stream(tg_str);
    while (std::getline(tg_stream, item, ',')) { tg_values.push_back(std::stoi(item)); }

    // Check that pp and tg have the same number of items
    MLLM_RT_ASSERT_EQ(pp_values.size(), tg_values.size());

    // Create pairs
    for (size_t i = 0; i < pp_values.size(); ++i) { pp_tg_pairs.emplace_back(pp_values[i], tg_values[i]); }
  }

  // Actual run for configurable number of turns
  mllm::print("\n========================================");
  mllm::print("Starting Benchmark Tests");
  mllm::print("========================================\n");

  for (auto [pp, tg] : pp_tg_pairs) {
    mllm::print("----------------------------------------");
    mllm::print("Test Configuration:");
    mllm::print("  Prompt Length (PP)    :", pp);
    mllm::print("  Generation Length (TG):", tg);
    mllm::print("----------------------------------------");

    std::vector<BenchmarkTemplateResult> results;
    results.reserve(static_cast<size_t>(R));

    for (int i = 0; i < R; ++i) {
      mllm::print("  Run", i + 1, "of", R, "...");

      benchmark->clear();
      auto result = benchmark->run(pp, tg);
      results.push_back(result);

      mllm::print("    TTFT         :", result.ttft, "ms");
      mllm::print("    Prefill Speed:", result.prefill_speed, "tokens/s");
      mllm::print("    Decode Speed :", result.decode_speed, "tokens/s");

      float prefill_ms = (result.prefill_speed > 0.0f) ? (pp / result.prefill_speed) * 1000.0f : 0.0f;
      float decode_ms_per_tok = (result.decode_speed > 0.0f) ? (1.0f / result.decode_speed) * 1000.0f : 0.0f;
      mllm::print("    Prefill Latency   :", prefill_ms, "ms");
      mllm::print("    Decode Latency    :", decode_ms_per_tok, "ms");

      int cool = cooldown_s.get();
      if (i + 1 < R && cool > 0) {
        mllm::print("    Cooling down for", cool, "seconds...");
        std::this_thread::sleep_for(std::chrono::seconds(cool));
      }
    }

    float denom = (R > 0) ? static_cast<float>(R) : 1.0f;
    float avg_ttft = 0.0f;
    float avg_prefill_speed = 0.0f;
    float avg_decode_speed = 0.0f;

    for (const auto& result : results) {
      avg_ttft += result.ttft;
      avg_prefill_speed += result.prefill_speed;
      avg_decode_speed += result.decode_speed;
    }

    avg_ttft /= 3.0f;
    avg_prefill_speed /= 3.0f;
    avg_decode_speed /= 3.0f;

    // Print average results
    mllm::print("\n========== Average Results ==========");
    mllm::print("Configuration: PP=", pp, " TG=", tg);
    mllm::print("Average TTFT         :", avg_ttft, "ms");
    mllm::print("Average Prefill Speed:", avg_prefill_speed, "tokens/s");
    mllm::print("Average Decode Speed :", avg_decode_speed, "tokens/s");
    mllm::print("=====================================\n");

    avg_ttft /= denom;
    avg_prefill_speed /= denom;
    avg_decode_speed /= denom;

    float avg_prefill_ms = (avg_prefill_speed > 0.0f) ? (pp / avg_prefill_speed) * 1000.0f : 0.0f;
    float avg_decode_ms_per_tok = (avg_decode_speed > 0.0f) ? (1.0f / avg_decode_speed) * 1000.0f : 0.0f;

    // KV cache estimate
    double kv_est_bytes_pp = 0.0;
    double kv_est_bytes_final = 0.0;
    if (auto info = benchmark->kvEstimateInfo(); info.has_value()) {
      const int32_t bytes_per = kv_dtype_bytes.get();  // 1/2/4
      // LLaMA-like KV: 2 * n_layers * n_kv_heads * head_dim * seq_len * bytes
      kv_est_bytes_pp = 2.0 * info->num_layers * info->num_kv_heads * info->head_dim * (double)pp * bytes_per;
      kv_est_bytes_final = 2.0 * info->num_layers * info->num_kv_heads * info->head_dim * (double)(pp + tg) * bytes_per;
    }

    std::stringstream ss;
    ss << schema_version.get() << "," << STRINGIFY(MLLM_GIT_COMMIT_HASH) << "," << mllm::cpu::CURRENT_ARCH_STRING << ","
       << model_name.get() << "," << cache_length.get() << "," << pp << "," << tg << "," << avg_ttft << "," << avg_prefill_speed << "," << avg_decode_speed
       << "," << avg_prefill_ms << "," << avg_decode_ms_per_tok << "," << kv_est_bytes_pp << "," << kv_est_bytes_final;

    if (csv_file.is_open()) { csv_file << ss.str() << std::endl; }
  }

  mllm::print("\n========================================");
  mllm::print("Benchmark Tests Completed");
  mllm::print("========================================");

  if (csv_file.is_open()) { csv_file.close(); }
})
