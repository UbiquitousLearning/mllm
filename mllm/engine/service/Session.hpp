// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "mllm/models/ARGeneration.hpp"

namespace mllm::service {

class Session {
 public:
  using ptr_t = std::shared_ptr<Session>;

  explicit Session(std::shared_ptr<mllm::models::ARGeneration> model);

  virtual void streamGenerate(const nlohmann::json& request,
                              const std::function<void(const nlohmann::json&, bool)>& callback) = 0;

  virtual void fromPreTrain(const std::string& model_path);

 private:
  std::shared_ptr<mllm::models::ARGeneration> model_;
};

class NoneSession final : public Session {
 public:
  NoneSession();

  void streamGenerate(const nlohmann::json& request, const std::function<void(const nlohmann::json&, bool)>& callback) override;
};

}  // namespace mllm::service
