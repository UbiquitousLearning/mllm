// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "mllm/backends/base/Backend.hpp"

namespace mllm::cuda {

class CudaBackend final : public Backend {
 public:
  CudaBackend();
};

std::shared_ptr<CudaBackend> createCudaBackend();
}  // namespace mllm::cuda
