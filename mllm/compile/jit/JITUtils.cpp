// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <bit>

#include "mllm/compile/jit/JITUtils.hpp"

namespace mllm::jit {

bool isLittleEndian() noexcept { return std::endian::native == std::endian::little; }

bool isBigEndian() noexcept { return std::endian::native == std::endian::big; }

}  // namespace mllm::jit
