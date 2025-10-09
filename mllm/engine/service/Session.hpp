// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <nlohmann/json.hpp>

namespace mllm::service {

class Session {
 public:
  using ptr_t = std::shared_ptr<Session>;

  Session() = default;

  virtual void streamGenerate(const nlohmann::json& request,
                              const std::function<void(const nlohmann::json&, bool)>& callback) = 0;

  virtual void fromPreTrain(const std::string& model_path);
};

class NoneSession final : public Session {
 public:
  NoneSession();

  void streamGenerate(const nlohmann::json& request, const std::function<void(const nlohmann::json&, bool)>& callback) override;
};

}  // namespace mllm::service
