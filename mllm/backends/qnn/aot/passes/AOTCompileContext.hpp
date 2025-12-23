// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <nlohmann/json.hpp>
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"

namespace mllm::qnn::aot {

class AOTCompileContext {
 public:
  // Get singleton instance
  static AOTCompileContext& getInstance() {
    static AOTCompileContext instance;
    return instance;
  }

  // Delete copy constructor and assignment operator
  AOTCompileContext(const AOTCompileContext&) = delete;
  AOTCompileContext& operator=(const AOTCompileContext&) = delete;

  // Accessors
  [[nodiscard]] QnnAOTEnv* getEnv() const { return env_; }

  void setEnv(QnnAOTEnv* env) { env_ = env; }

  void setConfig(const std::string& fp);

  nlohmann::json& getConfig();

 private:
  // Private constructor
  AOTCompileContext() = default;

  QnnAOTEnv* env_ = nullptr;
  nlohmann::json config_;
};

}  // namespace mllm::qnn::aot
