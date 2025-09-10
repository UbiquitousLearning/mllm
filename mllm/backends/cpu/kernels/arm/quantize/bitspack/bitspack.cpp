// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cmath>
#include "mllm/utils/Common.hpp"

#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/bitspack.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::arm::bitspack {

namespace quant_internal {

void get_qvals_range(int& qmin, int& qmax, int nbit, bool is_symmetric) {
  if (is_symmetric) {
    qmin = -(1 << (nbit - 1)) + 1;
    qmax = -qmin;
  } else {
    qmin = -(1 << (nbit - 1));
    qmax = (1 << (nbit - 1)) - 1;
  }
}

float get_scale(float vmin, float vmax, int qmin, int qmax) {
  assert(qmin < qmax);
  assert(vmin < vmax);
  return (vmax - vmin) / (qmax - qmin);
}

void get_scale_and_zero(float& scale, int& zero, float vmin, float vmax, int qmin, int qmax) {
  scale = get_scale(vmin, vmax, qmin, qmax);
  zero = qmin - std::round(vmin / scale);
}

void find_min_and_max(float& min, float& max, const float* vals, int size) {
  assert(size > 0);

  // Needed in case size < 4 so we don't compare to
  // uninitialized min/max values
  min = vals[0];
  max = min;

  int i = 0;
  if (i + 3 < size) {
    float32x4_t mins = vld1q_f32(vals + i);
    float32x4_t maxes = mins;
    i += 4;
    for (; i + 3 < size; i += 4) {
      float32x4_t v = vld1q_f32(vals + i);
      mins = vminq_f32(mins, v);
      maxes = vmaxq_f32(maxes, v);
    }
    min = vminvq_f32(mins);
    max = vmaxvq_f32(maxes);
  }

  // Remainder
  while (i < size) {
    if (vals[i] < min) { min = vals[i]; }
    if (vals[i] > max) { max = vals[i]; }
    i += 1;
  }
}

int32_t compute_sum(const int8_t* vals, int size) {
  assert(size >= 1);

  int32_t res = 0;
  int i = 0;

#pragma unroll(4)
  for (; i + 15 < size; i += 16) {
    int8x16_t vec_vals = vld1q_s8(vals + i);
    res += (int)(vaddlvq_s8(vec_vals));
  }
  for (; i < size; i += 1) { res += vals[i]; }
  return res;
}

namespace {
inline void _vec_clip_inplace(int32x4_t& vec, int32x4_t vec_min, int32x4_t vec_max) {
  vec = vmaxq_s32(vec, vec_min);
  vec = vminq_s32(vec, vec_max);
}

}  // namespace

void quantize(
    // Output
    int8_t* qvals,
    // Inputs
    const float32_t* vals, int size, float32_t scale, int8_t zero, int8_t qmin, int8_t qmax) {
  float32_t invScale = 1.0 / (scale + 1e-16);
  float32x4_t vec_zero = vdupq_n_f32(zero);
  float32x4_t vec_invScale = vdupq_n_f32(invScale);
  int32x4_t vec_qmin = vdupq_n_s32(qmin);
  int32x4_t vec_qmax = vdupq_n_s32(qmax);

  float32x4_t vec_val;
  float32x4_t vec_qval_f32;
  int32x4_t vec_qval_s32;
  int16x4_t vec_qval_s16_0;
  int16x4_t vec_qval_s16_1;

  int i = 0;
  for (; (i + 8) < size; i += 8) {
    //////////////////////////////////////
    // Quantize first 4 element chunk to int16
    //////////////////////////////////////
    vec_val = vld1q_f32(vals + i);

    // Quantize and round
    vec_qval_f32 = vfmaq_f32(vec_zero, vec_val, vec_invScale);
    vec_qval_s32 = vcvtnq_s32_f32(vec_qval_f32);

    _vec_clip_inplace(vec_qval_s32, vec_qmin, vec_qmax);

    vec_qval_s16_0 = vqmovn_s32(vec_qval_s32);

    //////////////////////////////////////
    // Quantize second 4 element chunk to int16
    //////////////////////////////////////
    vec_val = vld1q_f32(vals + i + 4);

    // Quantize and round
    vec_qval_f32 = vfmaq_f32(vec_zero, vec_val, vec_invScale);
    vec_qval_s32 = vcvtnq_s32_f32(vec_qval_f32);

    _vec_clip_inplace(vec_qval_s32, vec_qmin, vec_qmax);

    vec_qval_s16_1 = vqmovn_s32(vec_qval_s32);

    //////////////////////////////////////
    // Store 8 quantized elements
    //////////////////////////////////////
    int16x8_t vec_qval_s16_01 = vcombine_s16(vec_qval_s16_0, vec_qval_s16_1);
    int8x8_t vec_qval_s8_01 = vqmovn_s16(vec_qval_s16_01);
    vst1_s8(qvals + i, vec_qval_s8_01);
  }
  auto curr_rounding_mode = fegetround();
  fesetround(FE_TONEAREST);
  for (; i < size; ++i) {
    // Quantize remaining elements using scalar code
    float32_t val = vals[i];
    float32_t qval_f32 = zero + val * invScale;
    int32_t qval_s32 = static_cast<int32_t>(std::nearbyint(qval_f32));

    // Clip to qmin and qmax
    qval_s32 = std::max(static_cast<int32_t>(qmin), std::min(qval_s32, static_cast<int32_t>(qmax)));

    // Store the quantized value
    qvals[i] = static_cast<int8_t>(qval_s32);
  }
  fesetround(int(curr_rounding_mode));
}

}  // namespace quant_internal

namespace activation_packing {
MLLM_EMPTY_SCOPE;
}

namespace weight_packing {
MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::cpu::arm::bitspack

#endif
