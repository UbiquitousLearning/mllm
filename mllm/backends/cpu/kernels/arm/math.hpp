// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_fp16.h>
#endif

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#if !defined(MLLM_HOST_ARCH_ARM64) && !defined(MLLM_HOST_ARCH_ARM)
#error Arm compiler is required.
#else

#include <arm_neon.h>
#include <cmath>

namespace mllm::cpu::arm {

static inline float32x4_t vclampq_f32(float32x4_t x, float lo, float hi) {
  x = vminq_f32(x, vdupq_n_f32(hi));
  x = vmaxq_f32(x, vdupq_n_f32(lo));
  return x;
};

static inline float32x4_t vexpq_hp_f32(float32x4_t x) {
  float result[4];
  vst1q_f32(result, x);

#pragma unroll
  for (float& i : result) { i = expf(i); }

  return vld1q_f32(result);
}

// ref from ncnn.
//
// NEON implementation of exp
//
// Inspired by Intel Approximate Math library, and based on the
// corresponding algorithms of the cephes math library
//
// see:
// https://github.com/Tencent/ncnn/blob/master/src/layer/arm/neon_mathfun.h#L123
//
// Licence is below:
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f
#define c_cephes_LOG2EF 1.44269504088896341

#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

static inline float32x4_t vexpq_fast_f32(float32x4_t x) {
  float32x4_t tmp, fx;

  float32x4_t one = vdupq_n_f32(1);
  x = vclampq_f32(x, c_exp_lo, c_exp_hi);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  /* if greater, substract 1 */
  uint32x4_t mask = vcgtq_f32(tmp, fx);
  mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
  float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  z = vmulq_f32(x, x);

  float32x4_t y = vdupq_n_f32(c_cephes_exp_p0);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p1), y, x);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p2), y, x);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p3), y, x);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p4), y, x);
  y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p5), y, x);

  y = vmlaq_f32(x, y, z);
  y = vaddq_f32(y, one);

  /* build 2^n */
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
  mm = vshlq_n_s32(mm, 23);
  float32x4_t pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}

static inline float vsum_reduce_fp32(const float* __restrict__ X, int N) {
  float sum_ret = 0.0f;

  float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
  float32x4_t sum_vec1 = vdupq_n_f32(0.0f);

  int i = 0;
  for (; i <= N - 16; i += 16) {
    float32x4_t vec0 = vld1q_f32(X + i);
    float32x4_t vec1 = vld1q_f32(X + i + 4);
    float32x4_t vec2 = vld1q_f32(X + i + 8);
    float32x4_t vec3 = vld1q_f32(X + i + 12);

    sum_vec0 = vaddq_f32(sum_vec0, vec0);
    sum_vec0 = vaddq_f32(sum_vec0, vec1);
    sum_vec1 = vaddq_f32(sum_vec1, vec2);
    sum_vec1 = vaddq_f32(sum_vec1, vec3);
  }

  sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);

  for (; i <= N - 8; i += 8) {
    float32x4_t vec0 = vld1q_f32(X + i);
    float32x4_t vec1 = vld1q_f32(X + i + 4);

    sum_vec0 = vaddq_f32(sum_vec0, vec0);
    sum_vec0 = vaddq_f32(sum_vec0, vec1);
  }

  for (; i <= N - 4; i += 4) {
    float32x4_t vec0 = vld1q_f32(X + i);
    sum_vec0 = vaddq_f32(sum_vec0, vec0);
  }

  sum_ret += vaddvq_f32(sum_vec0);

  for (; i < N; ++i) { sum_ret += X[i]; }

  return sum_ret;
}

static inline float vmax_reduce_fp32(const float* __restrict__ X, int N) {
  float max_ret = X[0];

  float32x4_t max_vec0 = vdupq_n_f32(max_ret);
  float32x4_t max_vec1 = vdupq_n_f32(max_ret);

  int i;
  for (i = 0; i <= N - 16; i += 16) {
    // load
    float32x4_t vec0 = vld1q_f32(X + i);
    float32x4_t vec1 = vld1q_f32(X + i + 4);
    float32x4_t vec2 = vld1q_f32(X + i + 8);
    float32x4_t vec3 = vld1q_f32(X + i + 12);

    // compare
    max_vec0 = vmaxq_f32(max_vec0, vec0);
    max_vec0 = vmaxq_f32(max_vec0, vec1);
    max_vec1 = vmaxq_f32(max_vec1, vec2);
    max_vec1 = vmaxq_f32(max_vec1, vec3);
  }

  for (; i <= N - 8; i += 8) {
    // load
    float32x4_t vec0 = vld1q_f32(X + i);
    float32x4_t vec1 = vld1q_f32(X + i + 4);

    // compare
    max_vec0 = vmaxq_f32(max_vec0, vec0);
    max_vec1 = vmaxq_f32(max_vec1, vec1);
  }

  max_vec0 = vmaxq_f32(max_vec0, max_vec1);

  for (; i <= N - 4; i += 4) {
    // load
    float32x4_t vec0 = vld1q_f32(X + i);

    // compare
    max_vec0 = vmaxq_f32(max_vec0, vec0);
  }

  max_ret = fmax(max_ret, vmaxvq_f32(max_vec0));

  for (; i < N; ++i) { max_ret = fmax(max_ret, X[i]); }

  return max_ret;
}

static inline float vmin_reduce_fp32(const float* __restrict__ X, int N) {
  float min_ret = X[0];

  float32x4_t min_vec0 = vdupq_n_f32(min_ret);
  float32x4_t min_vec1 = vdupq_n_f32(min_ret);

  int i;
  for (i = 0; i <= N - 16; i += 16) {
    // load
    float32x4_t vec0 = vld1q_f32(X + i);
    float32x4_t vec1 = vld1q_f32(X + i + 4);
    float32x4_t vec2 = vld1q_f32(X + i + 8);
    float32x4_t vec3 = vld1q_f32(X + i + 12);

    // compare
    min_vec0 = vminq_f32(min_vec0, vec0);
    min_vec0 = vminq_f32(min_vec0, vec1);
    min_vec1 = vminq_f32(min_vec1, vec2);
    min_vec1 = vminq_f32(min_vec1, vec3);
  }

  for (; i <= N - 8; i += 8) {
    // load
    float32x4_t vec0 = vld1q_f32(X + i);
    float32x4_t vec1 = vld1q_f32(X + i + 4);

    // compare
    min_vec0 = vminq_f32(min_vec0, vec0);
    min_vec1 = vminq_f32(min_vec1, vec1);
  }

  min_vec0 = vminq_f32(min_vec0, min_vec1);

  for (; i <= N - 4; i += 4) {
    // load
    float32x4_t vec0 = vld1q_f32(X + i);

    // compare
    min_vec0 = vmaxq_f32(min_vec0, vec0);
  }

  min_ret = fmin(min_ret, vminvq_f32(min_vec0));

  for (; i < N; ++i) { min_ret = fmin(min_ret, X[i]); }

  return min_ret;
}

static inline float32x4_t vsigmoid_f32(float32x4_t x) {
  float32x4_t ones = vdupq_n_f32(1.f);
  x = vnegq_f32(x);
  x = vexpq_fast_f32(x);
  x = vaddq_f32(x, ones);
  float32x4_t out = vrecpeq_f32(x);
  out = vmulq_f32(vrecpsq_f32(x, out), out);
  return vmulq_f32(vrecpsq_f32(x, out), out);
}

// perform (x * x).sum() / x.size()
static inline float vsquare_mean_fp32(const float* __restrict X, int dim) {
  float32x4_t sum = vdupq_n_f32(0.0f);
  float32x4_t square;

  int i;
  for (i = 0; i <= dim - 4; i += 4) {
    float32x4_t vec = vld1q_f32(&X[i]);
    square = vmulq_f32(vec, vec);
    sum = vaddq_f32(sum, square);
  }

  float acc = vaddvq_f32(sum);
  for (; i < dim; ++i) { acc += X[i] * X[i]; }
  return acc / (float)dim;
}

// ref from ncnn.
//
// see:
// https://github.com/Tencent/ncnn/blob/master/src/layer/arm/neon_mathfun_tanh.h#L37
//
// Licence is below:
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#define c_tanh_tiny 1e-4f
#define c_tanh_hi 9.0f
// The monomial coefficients of the numerator polynomial (odd).
#define c_tanh_alpha_1 4.89352455891786e-3f
#define c_tanh_alpha_3 6.37261928875436e-4f
#define c_tanh_alpha_5 1.48572235717979e-5f
#define c_tanh_alpha_7 5.12229709037114e-8f
#define c_tanh_alpha_9 -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f
// The monomial coefficients of the denominator polynomial (even).
#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
static inline float32x4_t vtanh_fp32(float32x4_t x) {
  float32x4_t x2 = vabsq_f32(x);

  uint32x4_t tiny_mask = vcgeq_f32(x2, vdupq_n_f32(c_tanh_tiny));

  // clamp the inputs to the range [-9, 9] since anything outside
  // this range is -/+1.0f in single-precision.
  x2 = vreinterpretq_f32_u32(vbslq_u32(vcgeq_f32(vdupq_n_f32(c_tanh_hi), x2), vreinterpretq_u32_f32(x2),
                                       vreinterpretq_u32_f32(vdupq_n_f32(c_tanh_hi))));

  // since the polynomials are odd/even, we need x**2.
  float32x4_t z = vmulq_f32(x2, x2);

  // evaluate the numerator polynomial y.
  float32x4_t y = vdupq_n_f32(c_tanh_alpha_13);
  y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_11), y, z);
  y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_9), y, z);
  y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_7), y, z);
  y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_5), y, z);
  y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_3), y, z);
  y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_1), y, z);
  y = vmulq_f32(y, x2);

  // evaluate the denominator polynomial w.
  float32x4_t w = vdupq_n_f32(c_tanh_beta_6);
  w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_4), w, z);
  w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_2), w, z);
  w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_0), w, z);

  // divide the numerator by the denominator.
#if __aarch64__
  y = vdivq_f32(y, w);
#else
  y = div_ps(y, w);
#endif

  // reinstate the sign.
  y = vreinterpretq_f32_u32(vbslq_u32(vdupq_n_u32(1u << 31), vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));

  // when the argument is very small in magnitude it's more accurate to just return it.
  y = vreinterpretq_f32_u32(vbslq_u32(tiny_mask, vreinterpretq_u32_f32(y), vreinterpretq_u32_f32(x)));

  return y;
}

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else

static inline float16x8_t vclampq_f16(float16x8_t x, float lo, float hi) {
  x = vminq_f16(x, vdupq_n_f16(hi));
  x = vminq_f16(x, vdupq_n_f16(lo));
  return x;
};

static inline float16x8_t vexpq_hp_f16(float16x8_t x) {
  float result[8];
  float32x4_t result_hi = vcvt_f32_f16(vget_high_f16(x));
  float32x4_t result_lo = vcvt_f32_f16(vget_low_f16(x));

  vst1q_f32(result, result_hi);
  vst1q_f32(result + 4, result_lo);

#pragma unroll
  for (float& i : result) { i = expf(i); }

  result_hi = vld1q_f32(result);
  result_lo = vld1q_f32(result + 4);

  return vcombine_f16(vcvt_f16_f32(result_hi), vcvt_f16_f32(result_lo));
}

static inline float16x8_t vexpq_fast_f16(float16x8_t x) {
  float32x4_t result_hi = vcvt_f32_f16(vget_high_f16(x));
  float32x4_t result_lo = vcvt_f32_f16(vget_low_f16(x));

  result_hi = vexpq_fast_f32(result_hi);
  result_lo = vexpq_fast_f32(result_lo);

  return vcombine_f16(vcvt_f16_f32(result_lo), vcvt_f16_f32(result_hi));
}

static inline float vsquare_mean_fp16(const float16_t* __restrict X, int dim) {
  float32x4_t sum = vdupq_n_f32(0.0f);

  int i;
  for (i = 0; i <= dim - 8; i += 8) {
    float16x8_t vec = vld1q_f16(X + i);
    float16x8_t square_f16 = vmulq_f16(vec, vec);

    sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(square_f16)));
    sum = vaddq_f32(sum, vcvt_f32_f16(vget_high_f16(square_f16)));
  }

  float acc = vaddvq_f32(sum);

  for (; i < dim; ++i) { acc += (float)(X[i] * X[i]); }

  return acc / (float)dim;
}

static inline float16x8_t vsigmoid_f16(float16x8_t x) {
  float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x));
  float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));

  float32x4_t ones = vdupq_n_f32(1.0f);
  x_low = vnegq_f32(x_low);
  x_low = vexpq_fast_f32(x_low);
  x_low = vaddq_f32(x_low, ones);
  float32x4_t out_low = vrecpeq_f32(x_low);
  out_low = vmulq_f32(vrecpsq_f32(x_low, out_low), out_low);
  out_low = vmulq_f32(vrecpsq_f32(x_low, out_low), out_low);

  x_high = vnegq_f32(x_high);
  x_high = vexpq_fast_f32(x_high);
  x_high = vaddq_f32(x_high, ones);
  float32x4_t out_high = vrecpeq_f32(x_high);
  out_high = vmulq_f32(vrecpsq_f32(x_high, out_high), out_high);
  out_high = vmulq_f32(vrecpsq_f32(x_high, out_high), out_high);

  return vcombine_f16(vcvt_f16_f32(out_low), vcvt_f16_f32(out_high));
}

#endif

}  // namespace mllm::cpu::arm
#endif

#endif