/**
 * @file u1.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-28
 *
 */
#pragma once

#include <cstdint>

#include "mllm/backends/cpu/kernels/arm/macro.hpp"

namespace mllm::cpu::arm {

// We use the code from torch for u1-u7 bits packing/unpacking.

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
MLLM_CPU_ARM_FORCE_INLINE void pack_8_uint1_values(uint8_t* packed, const uint8_t* unpacked) {
  // Input is 8 bytes
  // Output is 1 bytes
  packed[0] = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) { packed[0] |= (unpacked[i] << (7 - i)); }
}

MLLM_CPU_ARM_FORCE_INLINE void unpack_8_uint1_values(uint8_t* unpacked, const uint8_t* packed) {
// Input is 8 bits = 1 byte
// Output is 8 bytes
#pragma unroll
  for (int i = 0; i < 8; i++) { unpacked[i] = (packed[0] >> (7 - i)) & 1; }
}

}  // namespace mllm::cpu::arm
