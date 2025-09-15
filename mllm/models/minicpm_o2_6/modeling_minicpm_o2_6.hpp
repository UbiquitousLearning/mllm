// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/llama/modeling_llama.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpm_o2_6.hpp"
#include "mllm/models/vocos/modeling_vocos.hpp"
#include "mllm/utils/Common.hpp"
#include <cstdint>
#include <functional>
#include <cstdlib>
#include <ctime>

namespace mllm::models::minicpm_o2_6 {
class MiniCPMO2_6 : public ARGeneration {
  MiniCPMO2_6Config cfg;

  vocos::Vocos vocos_;

 public:
  explicit MiniCPMO2_6(const MiniCPMO2_6Config& cfg) : cfg(cfg) { vocos_.from_pretrained(""); }

  void streaming_prefill() { NYI("not implement"); }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; };
  void streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                      const std::function<void(int64_t)>& callback) override {
    NYI("not implement");
  }
};
}  // namespace mllm::models::minicpm_o2_6