// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "mllm/backends/base/Backend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

class OpenCLBackend final : public Backend {
 public:
  explicit OpenCLBackend();
  ~OpenCLBackend() = default;

  [[nodiscard]] std::shared_ptr<OpenCLRuntime> runtime() const { return runtime_; }

 private:
  std::shared_ptr<OpenCLRuntime> runtime_;
};

}  // namespace mllm::opencl
