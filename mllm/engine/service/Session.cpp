// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <string>
#include <thread>
#include <chrono>

#include "mllm/engine/service/Session.hpp"

namespace mllm::service {

Session::Session(std::shared_ptr<mllm::models::ARGeneration> model) : model_(std::move(model)) {}

NoneSession::NoneSession() : Session(nullptr) {}

void NoneSession::streamGenerate(const nlohmann::json& request,
                                 const std::function<void(const nlohmann::json&, bool)>& callback) {
  MLLM_ERROR("Fall back in None Session. No model loaded.");
  for (int i = 0; i < 10; ++i) {
    callback(std::to_string(i), false);
    std::this_thread::sleep_for(std::chrono::milliseconds(i * 100));
  }
  callback("", true);
}

}  // namespace mllm::service
