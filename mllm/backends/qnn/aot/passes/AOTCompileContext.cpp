// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

void AOTCompileContext::setConfig(const std::string& fp) {
  std::ifstream ifs(fp);
  if (!ifs.is_open()) { throw std::runtime_error("Failed to open config file: " + fp); }
  try {
    ifs >> config_;
  } catch (const nlohmann::json::parse_error& e) {
    throw std::runtime_error(std::string("Failed to parse JSON config: ") + e.what());
  }
}

nlohmann::json& AOTCompileContext::getConfig() { return config_; }

}  // namespace mllm::qnn::aot
