// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace mllm::async::io {

enum class AsyncIOTaskTypes : int32_t {
  kCopy = 4096,
  kPromoteMMAPTensor2AnonymousMemoryTensor = 4097,
  kLoadAnonymousMemoryTensorFromDisk = 4098,
};

}
