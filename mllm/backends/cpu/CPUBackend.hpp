// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "mllm/backends/base/Backend.hpp"

namespace mllm::cpu {

class CPUBackend final : public Backend {
 public:
  CPUBackend();
};

std::shared_ptr<CPUBackend> createCPUBackend();
}  // namespace mllm::cpu
