// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mllm::models::qwen_npu {

struct ShadowNPUHeadSchedule {
  std::vector<std::vector<float>> retentions;
  float average_retention = 0.0F;
  int32_t dense_heads = 0;
  std::string normalized_dense_layers = "none";
};

inline std::string shadowNPUDenseLayerRanges(const std::vector<std::vector<float>>& retentions) {
  std::string result;
  int32_t run_begin = -1;
  const auto append_run = [](int32_t begin, int32_t end, std::string& output) {
    if (!output.empty()) { output += ','; }
    output += std::to_string(begin);
    if (end != begin) { output += '-' + std::to_string(end); }
  };

  for (int32_t layer = 0; layer <= static_cast<int32_t>(retentions.size()); ++layer) {
    const bool dense =
        layer < static_cast<int32_t>(retentions.size()) && !retentions[layer].empty()
        && std::all_of(retentions[layer].begin(), retentions[layer].end(), [](float retention) { return retention == 1.0F; });
    if (dense && run_begin < 0) { run_begin = layer; }
    if (!dense && run_begin >= 0) {
      append_run(run_begin, layer - 1, result);
      run_begin = -1;
    }
  }
  return result.empty() ? "none" : result;
}

inline ShadowNPUHeadSchedule parseShadowNPUHeadSchedule(std::istream& input, const std::string& source, int32_t layers,
                                                        int32_t heads) {
  if (layers <= 0 || heads <= 0) { throw std::invalid_argument("ShadowNPU head schedule requires positive layer/head counts"); }

  ShadowNPUHeadSchedule schedule;
  schedule.retentions.assign(layers, std::vector<float>(heads));
  double sum = 0.0;
  for (int32_t layer = 0; layer < layers; ++layer) {
    for (int32_t head = 0; head < heads; ++head) {
      float retention = 0.0F;
      if (!(input >> retention)) {
        throw std::invalid_argument("ShadowNPU profile has fewer entries than the model: " + source);
      }
      if (!(retention > 0.0F && retention <= 1.0F) || !std::isfinite(retention)) {
        throw std::invalid_argument("ShadowNPU head retention must be finite and in (0, 1]");
      }
      schedule.retentions[layer][head] = retention;
      sum += retention;
      if (retention == 1.0F) { ++schedule.dense_heads; }
    }
  }

  input >> std::ws;
  if (!input.eof()) { throw std::invalid_argument("ShadowNPU profile has more entries than the model: " + source); }

  schedule.average_retention = static_cast<float>(sum / static_cast<double>(layers * heads));
  schedule.normalized_dense_layers = shadowNPUDenseLayerRanges(schedule.retentions);
  return schedule;
}

inline ShadowNPUHeadSchedule loadShadowNPUHeadSchedule(const std::string& path, int32_t layers, int32_t heads) {
  std::ifstream input(path);
  if (!input) { throw std::invalid_argument("cannot open ShadowNPU head-retention profile: " + path); }
  return parseShadowNPUHeadSchedule(input, path, layers, heads);
}

}  // namespace mllm::models::qwen_npu
