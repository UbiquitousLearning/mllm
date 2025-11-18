// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "mllm/backends/base/Backend.hpp"

namespace mllm::ascend {

class AscendBackend final : public Backend {
 public:
  AscendBackend();
};

std::shared_ptr<AscendBackend> createAscendBackend();
}  // namespace mllm::ascend
