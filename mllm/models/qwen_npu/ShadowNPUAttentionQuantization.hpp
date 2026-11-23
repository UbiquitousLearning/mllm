// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "mllm/core/DataTypes.hpp"

namespace mllm::models::qwen_npu {

template <bool RoundThroughHalf = false>
inline void quantizeShadowAttentionInt8(const float* input, int8_t* output, size_t count, float inverse_scale) {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
  size_t i = 0;
#if defined(__aarch64__)
  const auto half = vdupq_n_f32(0.5F);
  const auto rounded = [inverse_scale, half](const float* values) {
    auto input_values = vld1q_f32(values);
    if constexpr (RoundThroughHalf) { input_values = vcvt_f32_f16(vcvt_f16_f32(input_values)); }
    auto scaled = vmulq_n_f32(input_values, inverse_scale);
    // The Android fast-math build can contract separate NEON intrinsics into FMLA.
    // Keep the intermediate FP32 rounding without a memory round trip.
    asm("" : "+w"(scaled));
    return vcvtmq_s32_f32(vaddq_f32(scaled, half));
  };
  for (; count - i >= 16; i += 16) {
    const auto a = rounded(input + i);
    const auto b = rounded(input + i + 4);
    const auto c = rounded(input + i + 8);
    const auto d = rounded(input + i + 12);
    const auto low = vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)));
    const auto high = vqmovn_s16(vcombine_s16(vqmovn_s32(c), vqmovn_s32(d)));
    vst1q_s8(output + i, vcombine_s8(low, high));
  }
#endif
  for (; i < count; ++i) {
    const auto bits = std::bit_cast<uint32_t>(input[i]);
    const auto magnitude = bits & 0x7fffffffU;
    if (magnitude >= 0x7f800000U) {
      output[i] = magnitude > 0x7f800000U ? 0 : (bits >> 31 ? -128 : 127);
      continue;
    }
    // Preserve the separate FP32 multiply and addition used by the tested encoder.
    float value = input[i];
    if constexpr (RoundThroughHalf) { value = static_cast<float>(static_cast<mllm_fp16_t>(value)); }
    const volatile float scaled = value * inverse_scale;
    const float shifted = scaled + 0.5F;
    output[i] = static_cast<int8_t>(std::floor(std::clamp(shifted, -128.0F, 127.0F)));
  }
}

}  // namespace mllm::models::qwen_npu
