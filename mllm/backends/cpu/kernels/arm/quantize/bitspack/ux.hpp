/**
 * @file ux.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-28
 *
 */
#pragma once

// We use the code from torch for u1-u7 bits packing/unpacking.

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/u1.hpp"
#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/u2.hpp"
#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/u3.hpp"
#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/u4.hpp"
#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/u5.hpp"
#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/u6.hpp"
#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/u7.hpp"

namespace mllm::cpu::arm::bitspack {

MLLM_CPU_ARM_FORCE_INLINE void vec_store_32_uint8_values(uint8_t* dest, const uint8x8_t& vec0, const uint8x8_t& vec1,
                                                         const uint8x8_t& vec2, const uint8x8_t& vec3) {
  vst1_u8(dest, vec0);
  vst1_u8(dest + 8, vec1);
  vst1_u8(dest + 16, vec2);
  vst1_u8(dest + 24, vec3);
}

MLLM_CPU_ARM_FORCE_INLINE void vec_load_32_uint8_values(uint8x8_t& vec0, uint8x8_t& vec1, uint8x8_t& vec2, uint8x8_t& vec3,
                                                        const uint8_t* src) {
  vec0 = vld1_u8(src);
  vec1 = vld1_u8(src + 8);
  vec2 = vld1_u8(src + 16);
  vec3 = vld1_u8(src + 24);
}

MLLM_CPU_ARM_FORCE_INLINE void vec_store_64_uint8_values(uint8_t* dest, const uint8x16_t& vec0, const uint8x16_t& vec1,
                                                         const uint8x16_t& vec2, const uint8x16_t& vec3) {
  vst1q_u8(dest, vec0);
  vst1q_u8(dest + 16, vec1);
  vst1q_u8(dest + 32, vec2);
  vst1q_u8(dest + 48, vec3);
}

MLLM_CPU_ARM_FORCE_INLINE void vec_load_64_uint8_values(uint8x16_t& vec0, uint8x16_t& vec1, uint8x16_t& vec2, uint8x16_t& vec3,
                                                        const uint8_t* src) {
  vec0 = vld1q_u8(src);
  vec1 = vld1q_u8(src + 16);
  vec2 = vld1q_u8(src + 32);
  vec3 = vld1q_u8(src + 48);
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_pack_32_lowbit_values(uint8_t* packed, const int8x16_t& unpacked0,
                                                         const int8x16_t& unpacked1) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  // Shift unpacked values to nonnegative range for quantization of 1-7 bits
  // No shifting is needed for 8-bit packing
  uint8x16_t shifted0;
  uint8x16_t shifted1;
  if constexpr (nbit < 8) {
    int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
    shifted0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
    shifted1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));
  }

  switch (nbit) {
    case 1:
      uint8_t buffer1[32];
      vst1q_u8(buffer1, shifted0);
      vst1q_u8(buffer1 + 16, shifted1);

      pack_8_uint1_values(packed, buffer1);
      pack_8_uint1_values(packed + 1, buffer1 + 8);
      pack_8_uint1_values(packed + 2, buffer1 + 16);
      pack_8_uint1_values(packed + 3, buffer1 + 24);
      break;
    case 2:
      vec_pack_32_uint2_values(packed, vget_low_u8(shifted0), vget_high_u8(shifted0), vget_low_u8(shifted1),
                               vget_high_u8(shifted1));
      break;
    case 3:
      uint8_t buffer3[32];
      vst1q_u8(buffer3, shifted0);
      vst1q_u8(buffer3 + 16, shifted1);

      pack_8_uint3_values(packed, buffer3);
      pack_8_uint3_values(packed + 3, buffer3 + 8);
      pack_8_uint3_values(packed + 6, buffer3 + 16);
      pack_8_uint3_values(packed + 9, buffer3 + 24);
      break;
    case 4: vec_pack_32_uint4_values(packed, shifted0, shifted1); break;
    case 5:
      uint8_t buffer5[32];
      vst1q_u8(buffer5, shifted0);
      vst1q_u8(buffer5 + 16, shifted1);

      pack_8_uint5_values(packed, buffer5);
      pack_8_uint5_values(packed + 5, buffer5 + 8);
      pack_8_uint5_values(packed + 10, buffer5 + 16);
      pack_8_uint5_values(packed + 15, buffer5 + 24);
      break;
    case 6: vec_pack_32_uint6_values(packed, shifted0, shifted1); break;
    case 7:
      uint8_t buffer7[32];
      vst1q_u8(buffer7, shifted0);
      vst1q_u8(buffer7 + 16, shifted1);

      pack_8_uint7_values(packed, buffer7);
      pack_8_uint7_values(packed + 7, buffer7 + 8);
      pack_8_uint7_values(packed + 14, buffer7 + 16);
      pack_8_uint7_values(packed + 21, buffer7 + 24);
      break;
    case 8:
      vst1q_u8(packed, vreinterpretq_u8_s8(unpacked0));
      vst1q_u8(packed + 16, vreinterpretq_u8_s8(unpacked1));
      break;
    default: assert(false);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_pack_32_uintx_values(uint8_t* packed, const uint8x16_t& unpacked0,
                                                        const uint8x16_t& unpacked1) {
  // Ensure nbit is within the valid range [1, 8]
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  switch (nbit) {
    case 1: {
      // For 1-bit, we store the 32 values into a temporary buffer
      // and then pack them in 8-value chunks.
      uint8_t buffer[32];
      vst1q_u8(buffer, unpacked0);
      vst1q_u8(buffer + 16, unpacked1);

      pack_8_uint1_values(packed, buffer);
      pack_8_uint1_values(packed + 1, buffer + 8);
      pack_8_uint1_values(packed + 2, buffer + 16);
      pack_8_uint1_values(packed + 3, buffer + 24);
      break;
    }
    case 2:
      // Use the existing vectorized implementation for 2-bit packing.
      vec_pack_32_uint2_values(packed, vget_low_u8(unpacked0), vget_high_u8(unpacked0), vget_low_u8(unpacked1),
                               vget_high_u8(unpacked1));
      break;
    case 3: {
      // For 3-bit, we store to a buffer and pack in 8-value chunks.
      uint8_t buffer[32];
      vst1q_u8(buffer, unpacked0);
      vst1q_u8(buffer + 16, unpacked1);

      pack_8_uint3_values(packed, buffer);
      pack_8_uint3_values(packed + 3, buffer + 8);
      pack_8_uint3_values(packed + 6, buffer + 16);
      pack_8_uint3_values(packed + 9, buffer + 24);
      break;
    }
    case 4:
      // Use the existing vectorized implementation for 4-bit packing.
      vec_pack_32_uint4_values(packed, unpacked0, unpacked1);
      break;
    case 5: {
      // For 5-bit, we store to a buffer and pack in 8-value chunks.
      uint8_t buffer[32];
      vst1q_u8(buffer, unpacked0);
      vst1q_u8(buffer + 16, unpacked1);

      pack_8_uint5_values(packed, buffer);
      pack_8_uint5_values(packed + 5, buffer + 8);
      pack_8_uint5_values(packed + 10, buffer + 16);
      pack_8_uint5_values(packed + 15, buffer + 24);
      break;
    }
    case 6:
      // Use the existing vectorized implementation for 6-bit packing.
      vec_pack_32_uint6_values(packed, unpacked0, unpacked1);
      break;
    case 7: {
      // For 7-bit, we store to a buffer and pack in 8-value chunks.
      uint8_t buffer[32];
      vst1q_u8(buffer, unpacked0);
      vst1q_u8(buffer + 16, unpacked1);

      pack_8_uint7_values(packed, buffer);
      pack_8_uint7_values(packed + 7, buffer + 8);
      pack_8_uint7_values(packed + 14, buffer + 16);
      pack_8_uint7_values(packed + 21, buffer + 24);
      break;
    }
    case 8:
      // For 8-bit, it's a direct memory store of the two vectors.
      vst1q_u8(packed, unpacked0);
      vst1q_u8(packed + 16, unpacked1);
      break;
    default:
      // This should be unreachable due to the static_asserts
      assert(false);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_32_lowbit_values(int8x16_t& unpacked0, int8x16_t& unpacked1, const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  uint8x16_t shifted0;
  uint8x16_t shifted1;

  switch (nbit) {
    case 1:
      uint8_t buffer1[32];
      unpack_8_uint1_values(buffer1, packed);
      unpack_8_uint1_values(buffer1 + 8, packed + 1);
      unpack_8_uint1_values(buffer1 + 16, packed + 2);
      unpack_8_uint1_values(buffer1 + 24, packed + 3);
      shifted0 = vld1q_u8(buffer1);
      shifted1 = vld1q_u8(buffer1 + 16);
      break;
    case 2:
      uint8x8_t shifted0_low;
      uint8x8_t shifted0_high;
      uint8x8_t shifted1_low;
      uint8x8_t shifted1_high;
      vec_unpack_32_uint2_values(shifted0_low, shifted0_high, shifted1_low, shifted1_high, packed);
      shifted0 = vcombine_u8(shifted0_low, shifted0_high);
      shifted1 = vcombine_u8(shifted1_low, shifted1_high);
      break;
    case 3:
      uint8_t buffer3[32];
      unpack_8_uint3_values(buffer3, packed);
      unpack_8_uint3_values(buffer3 + 8, packed + 3);
      unpack_8_uint3_values(buffer3 + 16, packed + 6);
      unpack_8_uint3_values(buffer3 + 24, packed + 9);
      shifted0 = vld1q_u8(buffer3);
      shifted1 = vld1q_u8(buffer3 + 16);
      break;
    case 4: vec_unpack_32_uint4_values(shifted0, shifted1, packed); break;
    case 5:
      uint8_t buffer5[32];
      unpack_8_uint5_values(buffer5, packed);
      unpack_8_uint5_values(buffer5 + 8, packed + 5);
      unpack_8_uint5_values(buffer5 + 16, packed + 10);
      unpack_8_uint5_values(buffer5 + 24, packed + 15);
      shifted0 = vld1q_u8(buffer5);
      shifted1 = vld1q_u8(buffer5 + 16);
      break;
    case 6: vec_unpack_32_uint6_values(shifted0, shifted1, packed); break;
    case 7:
      uint8_t buffer7[32];
      unpack_8_uint7_values(buffer7, packed);
      unpack_8_uint7_values(buffer7 + 8, packed + 7);
      unpack_8_uint7_values(buffer7 + 16, packed + 14);
      unpack_8_uint7_values(buffer7 + 24, packed + 21);
      shifted0 = vld1q_u8(buffer7);
      shifted1 = vld1q_u8(buffer7 + 16);
      break;
    case 8:
      unpacked0 = vreinterpretq_s8_u8(vld1q_u8(packed));
      unpacked1 = vreinterpretq_s8_u8(vld1q_u8(packed + 16));
      break;
    default: assert(false);
  }

  // unshift to move unpacked values to full range
  // no shifting is needed for 8-bit packing
  if constexpr (nbit < 8) {
    int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
    unpacked0 = vaddq_s8(vreinterpretq_s8_u8(shifted0), unshift);
    unpacked1 = vaddq_s8(vreinterpretq_s8_u8(shifted1), unshift);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_pack_64_lowbit_values(uint8_t* packed, const int8x16_t& unpacked0,
                                                         const int8x16_t& unpacked1, const int8x16_t& unpacked2,
                                                         const int8x16_t& unpacked3) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  // Shift unpacked values to nonnegative range for quantization of 1-7 bits
  // No shifting is needed for 8-bit packing
  uint8x16_t shifted0;
  uint8x16_t shifted1;
  uint8x16_t shifted2;
  uint8x16_t shifted3;
  if constexpr (nbit < 8) {
    int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
    shifted0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
    shifted1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));
    shifted2 = vreinterpretq_u8_s8(vaddq_s8(unpacked2, shift));
    shifted3 = vreinterpretq_u8_s8(vaddq_s8(unpacked3, shift));
  }

  switch (nbit) {
    case 1: vec_pack_64_uint1_values(packed, shifted0, shifted1, shifted2, shifted3); break;
    case 2: vec_pack_64_uint2_values(packed, shifted0, shifted1, shifted2, shifted3); break;
    case 3: vec_pack_64_uint3_values(packed, shifted0, shifted1, shifted2, shifted3); break;
    case 4:
      vec_pack_32_uint4_values(packed, shifted0, shifted1);
      vec_pack_32_uint4_values(packed + 16, shifted2, shifted3);
      break;
    case 5: vec_pack_64_uint5_values(packed, shifted0, shifted1, shifted2, shifted3); break;
    case 6: vec_pack_64_uint6_values(packed, shifted0, shifted1, shifted2, shifted3); break;
    case 7: vec_pack_64_uint7_values(packed, shifted0, shifted1, shifted2, shifted3); break;
    case 8:
      vst1q_u8(packed, vreinterpretq_u8_s8(unpacked0));
      vst1q_u8(packed + 16, vreinterpretq_u8_s8(unpacked1));
      vst1q_u8(packed + 32, vreinterpretq_u8_s8(unpacked2));
      vst1q_u8(packed + 48, vreinterpretq_u8_s8(unpacked3));
      break;
    default: assert(false);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_pack_64_uintx_values(uint8_t* packed, const uint8x16_t& unpacked0,
                                                        const uint8x16_t& unpacked1, const uint8x16_t& unpacked2,
                                                        const uint8x16_t& unpacked3) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  // No shifting is needed because the data is already unsigned.

  switch (nbit) {
    case 1:
      // The internal functions are already designed to take uint8x16_t
      vec_pack_64_uint1_values(packed, unpacked0, unpacked1, unpacked2, unpacked3);
      break;
    case 2: vec_pack_64_uint2_values(packed, unpacked0, unpacked1, unpacked2, unpacked3); break;
    case 3: vec_pack_64_uint3_values(packed, unpacked0, unpacked1, unpacked2, unpacked3); break;
    case 4:
      vec_pack_32_uint4_values(packed, unpacked0, unpacked1);
      vec_pack_32_uint4_values(packed + 16, unpacked2, unpacked3);
      break;
    case 5: vec_pack_64_uint5_values(packed, unpacked0, unpacked1, unpacked2, unpacked3); break;
    case 6: vec_pack_64_uint6_values(packed, unpacked0, unpacked1, unpacked2, unpacked3); break;
    case 7: vec_pack_64_uint7_values(packed, unpacked0, unpacked1, unpacked2, unpacked3); break;
    case 8:
      vst1q_u8(packed, unpacked0);
      vst1q_u8(packed + 16, unpacked1);
      vst1q_u8(packed + 32, unpacked2);
      vst1q_u8(packed + 48, unpacked3);
      break;
    default: assert(false);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_64_lowbit_values(int8x16_t& unpacked0, int8x16_t& unpacked1, int8x16_t& unpacked2,
                                                           int8x16_t& unpacked3, const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  uint8x16_t shifted0;
  uint8x16_t shifted1;
  uint8x16_t shifted2;
  uint8x16_t shifted3;

  switch (nbit) {
    case 1: vec_unpack_64_uint1_values(shifted0, shifted1, shifted2, shifted3, packed); break;
    case 2: vec_unpack_64_uint2_values(shifted0, shifted1, shifted2, shifted3, packed); break;
    case 3: vec_unpack_64_uint3_values(shifted0, shifted1, shifted2, shifted3, packed); break;
    case 4:
      vec_unpack_32_uint4_values(shifted0, shifted1, packed);
      vec_unpack_32_uint4_values(shifted2, shifted3, packed + 16);
      break;
    case 5: vec_unpack_64_uint5_values(shifted0, shifted1, shifted2, shifted3, packed); break;
    case 6: vec_unpack_64_uint6_values(shifted0, shifted1, shifted2, shifted3, packed); break;
    case 7: vec_unpack_64_uint7_values(shifted0, shifted1, shifted2, shifted3, packed); break;
    case 8:
      unpacked0 = vreinterpretq_s8_u8(vld1q_u8(packed));
      unpacked1 = vreinterpretq_s8_u8(vld1q_u8(packed + 16));
      unpacked2 = vreinterpretq_s8_u8(vld1q_u8(packed + 32));
      unpacked3 = vreinterpretq_s8_u8(vld1q_u8(packed + 48));
      break;
    default: assert(false);
  }

  // unshift to move unpacked values to full range
  // no shifting is needed for 8-bit packing
  if constexpr (nbit < 8) {
    int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
    unpacked0 = vaddq_s8(vreinterpretq_s8_u8(shifted0), unshift);
    unpacked1 = vaddq_s8(vreinterpretq_s8_u8(shifted1), unshift);
    unpacked2 = vaddq_s8(vreinterpretq_s8_u8(shifted2), unshift);
    unpacked3 = vaddq_s8(vreinterpretq_s8_u8(shifted3), unshift);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_pack_128_uintx_values(uint8_t* packed, const uint8x16_t& unpacked0,
                                                         const uint8x16_t& unpacked1, const uint8x16_t& unpacked2,
                                                         const uint8x16_t& unpacked3, const uint8x16_t& unpacked4,
                                                         const uint8x16_t& unpacked5, const uint8x16_t& unpacked6,
                                                         const uint8x16_t& unpacked7) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);
  switch (nbit) {
    case 1:
      vec_pack_128_uint1_values(packed, unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7);
      break;
    case 2:
      vec_pack_64_uint2_values(packed, unpacked0, unpacked1, unpacked2, unpacked3);
      vec_pack_64_uint2_values(packed + 16, unpacked4, unpacked5, unpacked6, unpacked7);
      break;
    case 3:
      vec_pack_128_uint3_values(packed, unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7);
      break;
    case 4:
      vec_pack_32_uint4_values(packed, unpacked0, unpacked1);
      vec_pack_32_uint4_values(packed + 16, unpacked2, unpacked3);
      vec_pack_32_uint4_values(packed + 32, unpacked4, unpacked5);
      vec_pack_32_uint4_values(packed + 48, unpacked6, unpacked7);
      break;
    case 5:
      vec_pack_128_uint5_values(packed, unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7);
      break;
    case 6:
      vec_pack_64_uint6_values(packed, unpacked0, unpacked1, unpacked2, unpacked3);
      vec_pack_64_uint6_values(packed + 48, unpacked4, unpacked5, unpacked6, unpacked7);
      break;
    case 7:
      vec_pack_128_uint7_values(packed, unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7);
      break;
    case 8:
      vst1q_u8(packed, unpacked0);
      vst1q_u8(packed + 16, unpacked1);
      vst1q_u8(packed + 32, unpacked2);
      vst1q_u8(packed + 48, unpacked3);
      vst1q_u8(packed + 64, unpacked4);
      vst1q_u8(packed + 80, unpacked5);
      vst1q_u8(packed + 96, unpacked6);
      vst1q_u8(packed + 112, unpacked7);
      break;
    default: assert(false);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_128_uintx_values(uint8x16_t& unpacked0, uint8x16_t& unpacked1, uint8x16_t& unpacked2,
                                                           uint8x16_t& unpacked3, uint8x16_t& unpacked4, uint8x16_t& unpacked5,
                                                           uint8x16_t& unpacked6, uint8x16_t& unpacked7,
                                                           const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);
  switch (nbit) {
    case 1:
      vec_unpack_128_uint1_values(unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7,
                                  packed);
      break;
    case 2:
      vec_unpack_64_uint2_values(unpacked0, unpacked1, unpacked2, unpacked3, packed);
      vec_unpack_64_uint2_values(unpacked4, unpacked5, unpacked6, unpacked7, packed + 16);
      break;
    case 3:
      vec_unpack_128_uint3_values(unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7,
                                  packed);
      break;
    case 4:
      vec_unpack_32_uint4_values(unpacked0, unpacked1, packed);
      vec_unpack_32_uint4_values(unpacked2, unpacked3, packed + 16);
      vec_unpack_32_uint4_values(unpacked4, unpacked5, packed + 32);
      vec_unpack_32_uint4_values(unpacked6, unpacked7, packed + 48);
      break;
    case 5:
      vec_unpack_128_uint5_values(unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7,
                                  packed);
      break;
    case 6:
      vec_unpack_64_uint6_values(unpacked0, unpacked1, unpacked2, unpacked3, packed);
      vec_unpack_64_uint6_values(unpacked4, unpacked5, unpacked6, unpacked7, packed + 48);
      break;
    case 7:
      vec_unpack_128_uint7_values(unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7,
                                  packed);
      break;
    case 8:
      unpacked0 = vld1q_u8(packed);
      unpacked1 = vld1q_u8(packed + 16);
      unpacked2 = vld1q_u8(packed + 32);
      unpacked3 = vld1q_u8(packed + 48);
      unpacked4 = vld1q_u8(packed + 64);
      unpacked5 = vld1q_u8(packed + 80);
      unpacked6 = vld1q_u8(packed + 96);
      unpacked7 = vld1q_u8(packed + 112);
      break;
    default: assert(false);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_pack_128_lowbit_values(uint8_t* packed, const int8x16_t& unpacked0,
                                                          const int8x16_t& unpacked1, const int8x16_t& unpacked2,
                                                          const int8x16_t& unpacked3, const int8x16_t& unpacked4,
                                                          const int8x16_t& unpacked5, const int8x16_t& unpacked6,
                                                          const int8x16_t& unpacked7) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  // Shift unpacked values to nonnegative range for quantization of 1-7 bits
  // No shifting is needed for 8-bit packing
  uint8x16_t shifted0;
  uint8x16_t shifted1;
  uint8x16_t shifted2;
  uint8x16_t shifted3;
  uint8x16_t shifted4;
  uint8x16_t shifted5;
  uint8x16_t shifted6;
  uint8x16_t shifted7;
  if constexpr (nbit < 8) {
    int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
    shifted0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
    shifted1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));
    shifted2 = vreinterpretq_u8_s8(vaddq_s8(unpacked2, shift));
    shifted3 = vreinterpretq_u8_s8(vaddq_s8(unpacked3, shift));
    shifted4 = vreinterpretq_u8_s8(vaddq_s8(unpacked4, shift));
    shifted5 = vreinterpretq_u8_s8(vaddq_s8(unpacked5, shift));
    shifted6 = vreinterpretq_u8_s8(vaddq_s8(unpacked6, shift));
    shifted7 = vreinterpretq_u8_s8(vaddq_s8(unpacked7, shift));
  }

  switch (nbit) {
    case 1:
      vec_pack_128_uint1_values(packed, shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7);
      break;
    case 2:
      vec_pack_64_uint2_values(packed, shifted0, shifted1, shifted2, shifted3);
      vec_pack_64_uint2_values(packed + 16, shifted4, shifted5, shifted6, shifted7);
      break;
    case 3:
      vec_pack_128_uint3_values(packed, shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7);
      break;
    case 4:
      vec_pack_32_uint4_values(packed, shifted0, shifted1);
      vec_pack_32_uint4_values(packed + 16, shifted2, shifted3);
      vec_pack_32_uint4_values(packed + 32, shifted4, shifted5);
      vec_pack_32_uint4_values(packed + 48, shifted6, shifted7);
      break;
    case 5:
      vec_pack_128_uint5_values(packed, shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7);
      break;
    case 6:
      vec_pack_64_uint6_values(packed, shifted0, shifted1, shifted2, shifted3);
      vec_pack_64_uint6_values(packed + 48, shifted4, shifted5, shifted6, shifted7);
      break;
    case 7:
      vec_pack_128_uint7_values(packed, shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7);
      break;
    case 8:
      vst1q_u8(packed, vreinterpretq_u8_s8(unpacked0));
      vst1q_u8(packed + 16, vreinterpretq_u8_s8(unpacked1));
      vst1q_u8(packed + 32, vreinterpretq_u8_s8(unpacked2));
      vst1q_u8(packed + 48, vreinterpretq_u8_s8(unpacked3));
      vst1q_u8(packed + 64, vreinterpretq_u8_s8(unpacked4));
      vst1q_u8(packed + 80, vreinterpretq_u8_s8(unpacked5));
      vst1q_u8(packed + 96, vreinterpretq_u8_s8(unpacked6));
      vst1q_u8(packed + 112, vreinterpretq_u8_s8(unpacked7));
      break;
    default: assert(false);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_128_lowbit_values(int8x16_t& unpacked0, int8x16_t& unpacked1, int8x16_t& unpacked2,
                                                            int8x16_t& unpacked3, int8x16_t& unpacked4, int8x16_t& unpacked5,
                                                            int8x16_t& unpacked6, int8x16_t& unpacked7, const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  uint8x16_t shifted0;
  uint8x16_t shifted1;
  uint8x16_t shifted2;
  uint8x16_t shifted3;
  uint8x16_t shifted4;
  uint8x16_t shifted5;
  uint8x16_t shifted6;
  uint8x16_t shifted7;

  switch (nbit) {
    case 1:
      vec_unpack_128_uint1_values(shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7, packed);
      break;
    case 2:
      vec_unpack_64_uint2_values(shifted0, shifted1, shifted2, shifted3, packed);
      vec_unpack_64_uint2_values(shifted4, shifted5, shifted6, shifted7, packed + 16);
      break;
    case 3:
      vec_unpack_128_uint3_values(shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7, packed);
      break;
    case 4:
      vec_unpack_32_uint4_values(shifted0, shifted1, packed);
      vec_unpack_32_uint4_values(shifted2, shifted3, packed + 16);
      vec_unpack_32_uint4_values(shifted4, shifted5, packed + 32);
      vec_unpack_32_uint4_values(shifted6, shifted7, packed + 48);
      break;
    case 5:
      vec_unpack_128_uint5_values(shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7, packed);
      break;
    case 6:
      vec_unpack_64_uint6_values(shifted0, shifted1, shifted2, shifted3, packed);
      vec_unpack_64_uint6_values(shifted4, shifted5, shifted6, shifted7, packed + 48);
      break;
    case 7:
      vec_unpack_128_uint7_values(shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7, packed);
      break;
    case 8:
      unpacked0 = vreinterpretq_s8_u8(vld1q_u8(packed));
      unpacked1 = vreinterpretq_s8_u8(vld1q_u8(packed + 16));
      unpacked2 = vreinterpretq_s8_u8(vld1q_u8(packed + 32));
      unpacked3 = vreinterpretq_s8_u8(vld1q_u8(packed + 48));
      unpacked4 = vreinterpretq_s8_u8(vld1q_u8(packed + 64));
      unpacked5 = vreinterpretq_s8_u8(vld1q_u8(packed + 80));
      unpacked6 = vreinterpretq_s8_u8(vld1q_u8(packed + 96));
      unpacked7 = vreinterpretq_s8_u8(vld1q_u8(packed + 112));
      break;
    default: assert(false);
  }

  // unshift to move unpacked values to full range
  // no shifting is needed for 8-bit packing
  if constexpr (nbit < 8) {
    int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
    unpacked0 = vaddq_s8(vreinterpretq_s8_u8(shifted0), unshift);
    unpacked1 = vaddq_s8(vreinterpretq_s8_u8(shifted1), unshift);
    unpacked2 = vaddq_s8(vreinterpretq_s8_u8(shifted2), unshift);
    unpacked3 = vaddq_s8(vreinterpretq_s8_u8(shifted3), unshift);
    unpacked4 = vaddq_s8(vreinterpretq_s8_u8(shifted4), unshift);
    unpacked5 = vaddq_s8(vreinterpretq_s8_u8(shifted5), unshift);
    unpacked6 = vaddq_s8(vreinterpretq_s8_u8(shifted6), unshift);
    unpacked7 = vaddq_s8(vreinterpretq_s8_u8(shifted7), unshift);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_128_lowbit_values_with_lut(int8x16_t& unpacked0, int8x16_t& unpacked1,
                                                                     int8x16_t& unpacked2, int8x16_t& unpacked3,
                                                                     int8x16_t& unpacked4, int8x16_t& unpacked5,
                                                                     int8x16_t& unpacked6, int8x16_t& unpacked7,
                                                                     const uint8_t* packed, const int8x16_t& lut) {
  static_assert(nbit <= 4);
  static_assert(nbit >= 1);
  uint8x16_t idx0;
  uint8x16_t idx1;
  uint8x16_t idx2;
  uint8x16_t idx3;
  uint8x16_t idx4;
  uint8x16_t idx5;
  uint8x16_t idx6;
  uint8x16_t idx7;
  vec_unpack_128_uintx_values<nbit>(idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, packed);
  unpacked0 = vqtbl1q_s8(lut, idx0);
  unpacked1 = vqtbl1q_s8(lut, idx1);
  unpacked2 = vqtbl1q_s8(lut, idx2);
  unpacked3 = vqtbl1q_s8(lut, idx3);
  unpacked4 = vqtbl1q_s8(lut, idx4);
  unpacked5 = vqtbl1q_s8(lut, idx5);
  unpacked6 = vqtbl1q_s8(lut, idx6);
  unpacked7 = vqtbl1q_s8(lut, idx7);
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_64_uintx_values(uint8x16_t& unpacked0, uint8x16_t& unpacked1, uint8x16_t& unpacked2,
                                                          uint8x16_t& unpacked3, const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  switch (nbit) {
    case 1: vec_unpack_64_uint1_values(unpacked0, unpacked1, unpacked2, unpacked3, packed); break;
    case 2: vec_unpack_64_uint2_values(unpacked0, unpacked1, unpacked2, unpacked3, packed); break;
    case 3: vec_unpack_64_uint3_values(unpacked0, unpacked1, unpacked2, unpacked3, packed); break;
    case 4:
      vec_unpack_32_uint4_values(unpacked0, unpacked1, packed);
      vec_unpack_32_uint4_values(unpacked2, unpacked3, packed + 16);
      break;
    case 5: vec_unpack_64_uint5_values(unpacked0, unpacked1, unpacked2, unpacked3, packed); break;
    case 6: vec_unpack_64_uint6_values(unpacked0, unpacked1, unpacked2, unpacked3, packed); break;
    case 7: vec_unpack_64_uint7_values(unpacked0, unpacked1, unpacked2, unpacked3, packed); break;
    case 8:
      unpacked0 = vld1q_u8(packed);
      unpacked1 = vld1q_u8(packed + 16);
      unpacked2 = vld1q_u8(packed + 32);
      unpacked3 = vld1q_u8(packed + 48);
      break;
    default: assert(false);
  }
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_64_lut_indices(uint8x16_t& unpacked0, uint8x16_t& unpacked1, uint8x16_t& unpacked2,
                                                         uint8x16_t& unpacked3, const uint8_t* packed) {
  static_assert(nbit <= 8);
  static_assert(nbit >= 1);

  if constexpr (nbit == 8) {
    unpacked0 = vld1q_u8(packed + 0);
    unpacked1 = vld1q_u8(packed + 16);
    unpacked2 = vld1q_u8(packed + 32);
    unpacked3 = vld1q_u8(packed + 48);
    return;
  }

  vec_unpack_64_uintx_values<nbit>(unpacked0, unpacked1, unpacked2, unpacked3, packed);

  const uint8_t mask = (1 << nbit) - 1;
  uint8x16_t mask_vec = vdupq_n_u8(mask);

  unpacked0 = vandq_u8(unpacked0, mask_vec);
  unpacked1 = vandq_u8(unpacked1, mask_vec);
  unpacked2 = vandq_u8(unpacked2, mask_vec);
  unpacked3 = vandq_u8(unpacked3, mask_vec);
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_32_uintx_values(uint8x16_t& unpacked0, uint8x16_t& unpacked1, const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  uint8x16_t shifted0 = vdupq_n_u8(0);
  uint8x16_t shifted1 = vdupq_n_u8(0);

  switch (nbit) {
    case 1:
      uint8_t buffer1[32];
      unpack_8_uint1_values(buffer1, packed);
      unpack_8_uint1_values(buffer1 + 8, packed + 1);
      unpack_8_uint1_values(buffer1 + 16, packed + 2);
      unpack_8_uint1_values(buffer1 + 24, packed + 3);
      shifted0 = vld1q_u8(buffer1);
      shifted1 = vld1q_u8(buffer1 + 16);
      break;
    case 2:
      uint8x8_t shifted0_low;
      uint8x8_t shifted0_high;
      uint8x8_t shifted1_low;
      uint8x8_t shifted1_high;
      vec_unpack_32_uint2_values(shifted0_low, shifted0_high, shifted1_low, shifted1_high, packed);
      shifted0 = vcombine_u8(shifted0_low, shifted0_high);
      shifted1 = vcombine_u8(shifted1_low, shifted1_high);
      break;
    case 3:
      uint8_t buffer3[32];
      unpack_8_uint3_values(buffer3, packed);
      unpack_8_uint3_values(buffer3 + 8, packed + 3);
      unpack_8_uint3_values(buffer3 + 16, packed + 6);
      unpack_8_uint3_values(buffer3 + 24, packed + 9);
      shifted0 = vld1q_u8(buffer3);
      shifted1 = vld1q_u8(buffer3 + 16);
      break;
    case 4: vec_unpack_32_uint4_values(shifted0, shifted1, packed); break;
    case 5:
      uint8_t buffer5[32];
      unpack_8_uint5_values(buffer5, packed);
      unpack_8_uint5_values(buffer5 + 8, packed + 5);
      unpack_8_uint5_values(buffer5 + 16, packed + 10);
      unpack_8_uint5_values(buffer5 + 24, packed + 15);
      shifted0 = vld1q_u8(buffer5);
      shifted1 = vld1q_u8(buffer5 + 16);
      break;
    case 6: vec_unpack_32_uint6_values(shifted0, shifted1, packed); break;
    case 7:
      uint8_t buffer7[32];
      unpack_8_uint7_values(buffer7, packed);
      unpack_8_uint7_values(buffer7 + 8, packed + 7);
      unpack_8_uint7_values(buffer7 + 16, packed + 14);
      unpack_8_uint7_values(buffer7 + 24, packed + 21);
      shifted0 = vld1q_u8(buffer7);
      shifted1 = vld1q_u8(buffer7 + 16);
      break;
    case 8:
      shifted0 = vld1q_u8(packed);
      shifted1 = vld1q_u8(packed + 16);
      break;
    default: assert(false);
  }
  unpacked0 = shifted0;
  unpacked1 = shifted1;
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_32_lut_indices(uint8x16_t& unpacked0, uint8x16_t& unpacked1, const uint8_t* packed) {
  static_assert(nbit <= 8);
  static_assert(nbit >= 1);

  // For 8-bit, the data is already unpacked. Just load directly.
  if constexpr (nbit == 8) {
    unpacked0 = vld1q_u8(packed + 0);
    unpacked1 = vld1q_u8(packed + 16);
    return;
  }

  // 1. Call the internal helper to get the raw unpacked values.
  vec_unpack_32_uintx_values<nbit>(unpacked0, unpacked1, packed);

  // 2. Apply the bitmask to get the final, correct indices for a LUT.
  const uint8_t mask = (1 << nbit) - 1;
  uint8x16_t mask_vec = vdupq_n_u8(mask);

  unpacked0 = vandq_u8(unpacked0, mask_vec);
  unpacked1 = vandq_u8(unpacked1, mask_vec);
}

template<int nbit>
MLLM_CPU_ARM_FORCE_INLINE void vec_unpack_128_lut_indices(uint8x16_t& unpacked0, uint8x16_t& unpacked1, uint8x16_t& unpacked2,
                                                          uint8x16_t& unpacked3, uint8x16_t& unpacked4, uint8x16_t& unpacked5,
                                                          uint8x16_t& unpacked6, uint8x16_t& unpacked7, const uint8_t* packed) {
  // Unpacks 128 tightly packed n-bit values into 8-bit LUT indices using ARM
  // NEON. For n-bit < 8, this function first spreads the bits into bytes and
  // then applies a mask to zero out the unused upper bits, ensuring each index
  // is valid. For the n-bit == 8 case, it's a direct memory load, as no
  // unpacking is needed.

  static_assert(nbit <= 8);
  static_assert(nbit >= 1);

  // For 8-bit, the data is already unpacked. Just load directly.
  if constexpr (nbit == 8) {
    unpacked0 = vld1q_u8(packed + 0);
    unpacked1 = vld1q_u8(packed + 16);
    unpacked2 = vld1q_u8(packed + 32);
    unpacked3 = vld1q_u8(packed + 48);
    unpacked4 = vld1q_u8(packed + 64);
    unpacked5 = vld1q_u8(packed + 80);
    unpacked6 = vld1q_u8(packed + 96);
    unpacked7 = vld1q_u8(packed + 112);
    return;
  }

  vec_unpack_128_uintx_values<nbit>(unpacked0, unpacked1, unpacked2, unpacked3, unpacked4, unpacked5, unpacked6, unpacked7,
                                    packed);
  const uint8_t mask = (1 << nbit) - 1;
  uint8x16_t mask_vec = vdupq_n_u8(mask);

  unpacked0 = vandq_u8(unpacked0, mask_vec);
  unpacked1 = vandq_u8(unpacked1, mask_vec);
  unpacked2 = vandq_u8(unpacked2, mask_vec);
  unpacked3 = vandq_u8(unpacked3, mask_vec);
  unpacked4 = vandq_u8(unpacked4, mask_vec);
  unpacked5 = vandq_u8(unpacked5, mask_vec);
  unpacked6 = vandq_u8(unpacked6, mask_vec);
  unpacked7 = vandq_u8(unpacked7, mask_vec);
}

}  // namespace mllm::cpu::arm::bitspack
#endif
