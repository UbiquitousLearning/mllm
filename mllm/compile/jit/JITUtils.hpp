// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <type_traits>

namespace mllm::jit {

/**
 * @brief Check if the system is little endian
 * @return true if the system is little endian, false if big endian
 */
bool isLittleEndian() noexcept;

/**
 * @brief Check if the system is big endian
 * @return true if the system is big endian, false if little endian
 */
bool isBigEndian() noexcept;

}  // namespace mllm::jit
