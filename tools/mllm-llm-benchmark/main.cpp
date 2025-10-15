// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <vector>
#include <sstream>

#include <mllm/mllm.hpp>
#include <mllm/utils/Argparse.hpp>
#include <mllm/utils/CPUArchHelper.hpp>
#include <mllm/utils/PlatformRTHelper.hpp>

#include "models/All.hpp"

MLLM_MAIN({
  auto& help = mllm::Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_name = mllm::Argparse::add<std::string>("-n|--model_name").help("Model name");
  auto& model_path = mllm::Argparse::add<std::string>("-m|--model_path").help("Model path");
  auto& config_path = mllm::Argparse::add<std::string>("-c|--config_path").help("Config path");
  auto& num_threads = mllm::Argparse::add<uint32_t>("-t|--threads").help("Number of threads");
  auto& pp = mllm::Argparse::add<std::string>("-pp|--prompt_length").help("Prompt length");
  auto& tg = mllm::Argparse::add<std::string>("-tg|--test_generation_length").help("Test Generation length");
  mllm::Argparse::parse(argc, argv);

  // Print Build Version
  mllm::print("MLLM Build Version :", MLLM_GIT_COMMIT_HASH);

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

  // TODO Actual run for 3 turns and gives avg results. Each turn will sleep for 5 seconds to let the SoC or GPU/NPU cool down.
  for (auto [pp, tg] : pp_tg_pairs) {
    benchmark->clear();

    for (int i = 0; i < 3; ++i) {
      // TODO
      auto result = benchmark->run(pp, tg);

      // TODO Sleep some times.
    }
    // TODO Calculate avg and print results
  }
})
