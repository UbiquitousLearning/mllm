// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/AscendBackend.hpp"
#include "mllm/backends/ascend/AscendAllocator.hpp"
#include "mllm/core/DeviceTypes.hpp"

namespace mllm::ascend {

AscendBackend::AscendBackend() : Backend(kAscend, createAscendAllocator()) {}

std::shared_ptr<AscendBackend> createAscendBackend() { return std::make_shared<AscendBackend>(); }

}  // namespace mllm::ascend
