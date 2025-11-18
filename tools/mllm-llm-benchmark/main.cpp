// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <vector>
#include <sstream>
#include <thread>
#include <chrono>

#include <mllm/mllm.hpp>
#include <mllm/utils/Argparse.hpp>
#include <mllm/utils/CPUArchHelper.hpp>
#include <mllm/utils/PlatformRTHelper.hpp>

#define STRINGIFY_INTERNAL(x) #x
#define STRINGIFY(x) STRINGIFY_INTERNAL(x)

#include "models/All.hpp"

MLLM_MAIN({
  auto& help = mllm::Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_name = mllm::Argparse::add<std::string>("-n|--model_name").help("Model name");
  auto& model_path = mllm::Argparse::add<std::string>("-m|--model_path").help("Model path");
  auto& config_path = mllm::Argparse::add<std::string>("-c|--config_path").help("Config path");
  auto& num_threads = mllm::Argparse::add<int32_t>("-t|--threads").help("Number of threads");
  auto& pp = mllm::Argparse::add<std::string>("-pp|--prompt_length").help("Prompt length");
  auto& tg = mllm::Argparse::add<std::string>("-tg|--test_generation_length").help("Test Generation length");
  auto& cache_length = mllm::Argparse::add<int32_t>("-cl|--cache_length").help("Cache length");
  mllm::Argparse::parse(argc, argv);

  // Print Build Version
  mllm::print("MLLM Build Version :", STRINGIFY(MLLM_GIT_COMMIT_HASH));

  // Print Device Info
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

  // Create benchmark
  mllm::print("Create Benchmark: ", model_name.get());
  auto benchmark = createBenchmark(model_name.get());
  MLLM_RT_ASSERT(benchmark != nullptr);

  // Print Model Info
  mllm::print("Model Info");
  benchmark->init(config_path.get(), model_path.get(), cache_length.get());
  benchmark->printModelInfo();

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

  // Actual run for 3 turns and gives avg results. Each turn will sleep for 5 seconds to let the SoC or GPU/NPU cool down.
  mllm::print("\n========================================");
  mllm::print("Starting Benchmark Tests");
  mllm::print("========================================\n");

  for (auto [pp, tg] : pp_tg_pairs) {
    mllm::print("----------------------------------------");
    mllm::print("Test Configuration:");
    mllm::print("  Prompt Length (PP)    :", pp);
    mllm::print("  Generation Length (TG):", tg);
    mllm::print("----------------------------------------");

    // Storage for results
    std::vector<BenchmarkTemplateResult> results;
    results.reserve(3);

    for (int i = 0; i < 3; ++i) {
      mllm::print("  Run", i + 1, "of 3...");

      // Clear cache before each run
      benchmark->clear();

      // Run benchmark
      auto result = benchmark->run(pp, tg);
      results.push_back(result);

      mllm::print("    TTFT         :", result.ttft, "ms");
      mllm::print("    Prefill Speed:", result.prefill_speed, "tokens/s");
      mllm::print("    Decode Speed :", result.decode_speed, "tokens/s");

      // Sleep for 5 seconds between runs to cool down
      if (i < 2) {
        mllm::print("    Cooling down for 5 seconds...");
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
    }

    // Calculate average results
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
  }

  mllm::print("\n========================================");
  mllm::print("Benchmark Tests Completed");
  mllm::print("========================================");
})
