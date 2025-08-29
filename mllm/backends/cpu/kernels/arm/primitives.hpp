// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>
#include <numbers>
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/utils/Common.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

#include "mllm/backends/cpu/kernels/arm/macro.hpp"

namespace mllm::cpu::arm {
template<typename __T>
struct __ScalarAdd {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a, __T b) { return a + b; }
};

template<typename __T>
struct __ScalarSub {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a, __T b) { return a - b; }
};

template<typename __T>
struct __ScalarMul {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a, __T b) { return a * b; }
};

template<typename __T>
struct __ScalarDiv {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a, __T b) { return a / b; }
};

template<typename __T>
struct __ScalarMax {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a, __T b) { return a > b ? a : b; }
};

template<typename __T>
struct __ScalarMin {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a, __T b) { return a < b ? a : b; }
};

template<typename __T>
struct __ScalarAbs {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a) { return std::abs(a); }
};

template<typename __T>
struct __ScalarLog {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a) { return std::log(a); }
};

template<typename __T>
struct __ScalarExp {
  static MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a) { return std::exp(a); }
};

template<typename __VT>
struct __VectorAdd {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a, __VT b) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorSub {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a, __VT b) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorMul {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a, __VT b) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorDiv {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a, __VT b) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorMax {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a, __VT b) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorMin {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a, __VT b) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorAbs {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorLog {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorExp {
  static MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a) {}

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

//===----------------------------------------------------------------------===//
// Int32 Vector Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorAdd<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t cal(int32x4_t a, int32x4_t b) { return vaddq_s32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorSub<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t cal(int32x4_t a, int32x4_t b) { return vsubq_s32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorMul<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t cal(int32x4_t a, int32x4_t b) { return vmulq_s32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorDiv<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t cal(int32x4_t a, int32x4_t b) {
    // Note: NEON does not have integer division, convert to float, divide, then convert back
    float32x4_t fa = vcvtq_f32_s32(a);
    float32x4_t fb = vcvtq_f32_s32(b);
    float32x4_t result = vdivq_f32(fa, fb);
    return vcvtq_s32_f32(result);
  }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorMax<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t cal(int32x4_t a, int32x4_t b) { return vmaxq_s32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorMin<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t cal(int32x4_t a, int32x4_t b) { return vminq_s32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorAbs<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t cal(int32x4_t a) { return vabsq_s32(a); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

//===----------------------------------------------------------------------===//
// Int16 Vector Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorAdd<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t cal(int16x8_t a, int16x8_t b) { return vaddq_s16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorSub<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t cal(int16x8_t a, int16x8_t b) { return vsubq_s16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorMul<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t cal(int16x8_t a, int16x8_t b) { return vmulq_s16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorDiv<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t cal(int16x8_t a, int16x8_t b) {
    // NEON does not have integer division, convert to float, divide, then convert back
    float32x4_t fa_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a)));
    float32x4_t fb_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b)));
    float32x4_t fa_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a)));
    float32x4_t fb_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b)));
    float32x4_t result_low = vdivq_f32(fa_low, fb_low);
    float32x4_t result_high = vdivq_f32(fa_high, fb_high);
    int16x8_t result;
    result = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(result_low)), vqmovn_s32(vcvtq_s32_f32(result_high)));
    return result;
  }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorMax<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t cal(int16x8_t a, int16x8_t b) { return vmaxq_s16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorMin<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t cal(int16x8_t a, int16x8_t b) { return vminq_s16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorAbs<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t cal(int16x8_t a) { return vabsq_s16(a); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

//===----------------------------------------------------------------------===//
// Int8 Vector Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorAdd<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t cal(int8x16_t a, int8x16_t b) { return vaddq_s8(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorSub<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t cal(int8x16_t a, int8x16_t b) { return vsubq_s8(a, b); }
  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorMul<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t cal(int8x16_t a, int8x16_t b) { return vmulq_s8(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorDiv<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t cal(int8x16_t a, int8x16_t b) {
    // NEON does not have integer division, convert to float, divide, then convert back
    int16x8_t a_low = vmovl_s8(vget_low_s8(a));
    int16x8_t b_low = vmovl_s8(vget_low_s8(b));
    int16x8_t a_high = vmovl_s8(vget_high_s8(a));
    int16x8_t b_high = vmovl_s8(vget_high_s8(b));

    float32x4_t fa_low_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_low)));
    float32x4_t fb_low_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_low)));
    float32x4_t fa_low_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_low)));
    float32x4_t fb_low_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_low)));
    float32x4_t fa_high_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_high)));
    float32x4_t fb_high_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_high)));
    float32x4_t fa_high_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_high)));
    float32x4_t fb_high_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_high)));

    float32x4_t result_low_low = vdivq_f32(fa_low_low, fb_low_low);
    float32x4_t result_low_high = vdivq_f32(fa_low_high, fb_low_high);
    float32x4_t result_high_low = vdivq_f32(fa_high_low, fb_high_low);
    float32x4_t result_high_high = vdivq_f32(fa_high_high, fb_high_high);

    int16x8_t result_low = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(result_low_low)), vqmovn_s32(vcvtq_s32_f32(result_low_high)));
    int16x8_t result_high =
        vcombine_s16(vqmovn_s32(vcvtq_s32_f32(result_high_low)), vqmovn_s32(vcvtq_s32_f32(result_high_high)));

    return vcombine_s8(vqmovn_s16(result_low), vqmovn_s16(result_high));
  }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorMax<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t cal(int8x16_t a, int8x16_t b) { return vmaxq_s8(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorMin<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t cal(int8x16_t a, int8x16_t b) { return vminq_s8(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorAbs<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t cal(int8x16_t a) { return vabsq_s8(a); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 16; }
};

//===----------------------------------------------------------------------===//
// Float32 Vector Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorAdd<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorSub<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a, float32x4_t b) { return vsubq_f32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorMul<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a, float32x4_t b) { return vmulq_f32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorDiv<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a, float32x4_t b) { return vdivq_f32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorMax<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a, float32x4_t b) { return vmaxq_f32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorMin<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a, float32x4_t b) { return vminq_f32(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorAbs<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a) { return vabsq_f32(a); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

// For Vector Log compute
namespace __mllm_vector_log {
static const float32x4_t one = vdupq_n_f32(1.0f);
static const float32x4_t ln2 = vdupq_n_f32(std::numbers::ln2_v<float>);
static const float32x4_t c1 = vdupq_n_f32(-0.5f);
static const float32x4_t c2 = vdupq_n_f32(0.3333333f);
static const float32x4_t c3 = vdupq_n_f32(-0.25f);
static const float32x4_t c4 = vdupq_n_f32(0.2f);

// FIXME:
// The code below is from LLM. Have precision errors.
inline float32x4_t log_ps_f32(float32x4_t a) {
  uint32x4_t xi = vreinterpretq_u32_f32(a);
  int32x4_t e = vreinterpretq_s32_u32(vshrq_n_u32(xi, 23));  // exponent bits
  e = vsubq_s32(e, vdupq_n_s32(127));                        // remove bias

  //  x / 2^e -> m ∈ [1,2)
  xi = vandq_u32(xi, vdupq_n_u32(0x007FFFFF));  // remain mantissa
  xi = vorrq_u32(xi, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)));
  float32x4_t m = vreinterpretq_f32_u32(xi);

  // m -> [0,1)，y = m - 1
  float32x4_t y = vsubq_f32(m, one);

  // log(1+y) = y - y^2/2 + y^3/3 - y^4/4 + ...
  float32x4_t p = y;
  float32x4_t y2 = vmulq_f32(y, y);

  float32x4_t poly = vmlaq_f32(c4, c3, y);  // c3*y + c4
  poly = vmlaq_f32(poly, c2, y);            // c2*y + ...
  poly = vmlaq_f32(poly, c1, y);            // c1*y + ...
  poly = vmulq_f32(poly, y2);
  poly = vaddq_f32(poly, y);

  // log(x) = e*ln(2) + poly
  float32x4_t ef = vcvtq_f32_s32(e);
  return vmlaq_f32(poly, ef, ln2);
}

}  // namespace __mllm_vector_log

template<>
struct __VectorLog<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a) { return __mllm_vector_log::log_ps_f32(a); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

namespace __mllm_vector_exp {
// from src/backends/cpu/compute/ActivationFunction.hpp in mllm:v1

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline float32x4_t exp_ps_f32(float32x4_t a) {
  const float32x4_t r = vdupq_n_f32(0x1.8p23f);
  const float32x4_t z = vfmaq_f32(r, a, vdupq_n_f32(std::numbers::log2e_v<float>));
  const float32x4_t n = vsubq_f32(z, r);
  const float32x4_t b = vfmsq_f32(vfmsq_f32(a, n, vdupq_n_f32(std::numbers::ln2_v<float>)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
  const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
  const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
  const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
  const float32x4_t u = vmulq_f32(b, b);
  const float32x4_t j = vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
                                  vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                                            vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u),
                                  u);
  if (!vpaddd_u64(vreinterpretq_u64_u32(c))) return vfmaq_f32(k, j, k);
  const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
  const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
  const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
  return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                   vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}
}  // namespace __mllm_vector_exp

template<>
struct __VectorExp<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a) { return __mllm_vector_exp::exp_ps_f32(a); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

//===----------------------------------------------------------------------===//
// Float16 Vector Ops
//===----------------------------------------------------------------------===//
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct __VectorAdd<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a, float16x8_t b) { return vaddq_f16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorSub<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a, float16x8_t b) { return vsubq_f16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorMul<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a, float16x8_t b) { return vmulq_f16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorDiv<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a, float16x8_t b) { return vdivq_f16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorMax<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a, float16x8_t b) { return vmaxq_f16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorMin<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a, float16x8_t b) { return vminq_f16(a, b); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorAbs<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a) { return vabsq_f16(a); }

  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorLog<float16x8_t> {
  static constexpr size_t lanes() { return 8; }
  static inline float16x8_t cal(float16x8_t a) {
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t hi = vcvt_f32_f16(vget_high_f16(a));

    lo = __mllm_vector_log::log_ps_f32(lo);
    hi = __mllm_vector_log::log_ps_f32(hi);

    return vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
  }
};

template<>
struct __VectorExp<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a) {
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t hi = vcvt_f32_f16(vget_high_f16(a));
    lo = __mllm_vector_exp::exp_ps_f32(lo);
    hi = __mllm_vector_exp::exp_ps_f32(hi);
    return vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
  }
  static MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

#endif  // #end if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

//===----------------------------------------------------------------------===//
// Scalar Load Ops
//===----------------------------------------------------------------------===//
template<typename T>
struct __ScalarLoad {
  static MLLM_CPU_ARM_FORCE_INLINE T load(const T* __restrict__ src, size_t pos) { return *(src + pos); }
};

//===----------------------------------------------------------------------===//
// Scalar Store Ops
//===----------------------------------------------------------------------===//
template<typename T>
struct __ScalarStore {
  static MLLM_CPU_ARM_FORCE_INLINE void store(T* __restrict__ dst, size_t pos, T value) { *(dst + pos) = value; }
};

//===----------------------------------------------------------------------===//
// Vector Load Ops
//===----------------------------------------------------------------------===//
template<typename T>
struct __VectorLoad {
  static MLLM_CPU_ARM_FORCE_INLINE T load(const typename T::type* __restrict__ src, size_t pos) {}
};

template<>
struct __VectorLoad<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t load(const int32_t* __restrict__ src, size_t pos) { return vld1q_s32(src + pos); }
  static MLLM_CPU_ARM_FORCE_INLINE int32x4_t load_scalar(const int32_t src) { return vdupq_n_s32(src); }
};

template<>
struct __VectorLoad<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t load(const int16_t* __restrict__ src, size_t pos) { return vld1q_s16(src + pos); }
  static MLLM_CPU_ARM_FORCE_INLINE int16x8_t load_scalar(const int16_t src) { return vdupq_n_s16(src); }
};

template<>
struct __VectorLoad<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t load(const int8_t* __restrict__ src, size_t pos) { return vld1q_s8(src + pos); }
  static MLLM_CPU_ARM_FORCE_INLINE int8x16_t load_scalar(const int8_t src) { return vdupq_n_s8(src); }
};

template<>
struct __VectorLoad<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t load(const float* __restrict__ src, size_t pos) { return vld1q_f32(src + pos); }
  static MLLM_CPU_ARM_FORCE_INLINE float32x4_t load_scalar(const float src) { return vdupq_n_f32(src); }
};

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct __VectorLoad<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t load(const float16_t* __restrict__ src, size_t pos) {
    return vld1q_f16(src + pos);
  }
  static MLLM_CPU_ARM_FORCE_INLINE float16x8_t load_scalar(const float16_t src) { return vdupq_n_f16(src); }
};
#endif

//===----------------------------------------------------------------------===//
// Vector Store Ops
//===----------------------------------------------------------------------===//
template<typename T>
struct __VectorStore {
  static MLLM_CPU_ARM_FORCE_INLINE void store(typename T::type* __restrict__ dst, size_t pos, T value) {}
};

template<>
struct __VectorStore<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE void store(int32_t* __restrict__ dst, size_t pos, int32x4_t value) {
    vst1q_s32(dst + pos, value);
  }
};

template<>
struct __VectorStore<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE void store(int16_t* __restrict__ dst, size_t pos, int16x8_t value) {
    vst1q_s16(dst + pos, value);
  }
};

template<>
struct __VectorStore<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE void store(int8_t* __restrict__ dst, size_t pos, int8x16_t value) {
    vst1q_s8(dst + pos, value);
  }
};

template<>
struct __VectorStore<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE void store(float* __restrict__ dst, size_t pos, float32x4_t value) {
    vst1q_f32(dst + pos, value);
  }
};

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct __VectorStore<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE void store(float16_t* __restrict__ dst, size_t pos, float16x8_t value) {
    vst1q_f16(dst + pos, value);
  }
};
#endif

//===----------------------------------------------------------------------===//
// Vector Reduce Ops
//===----------------------------------------------------------------------===//
template<typename __VT>
struct __VectorMaxReduce {
  static MLLM_CPU_ARM_FORCE_INLINE typename __VT::type cal(__VT v) {}
};

template<typename __VT>
struct __VectorSumReduce {
  static MLLM_CPU_ARM_FORCE_INLINE typename __VT::type cal(__VT v) {}
};

template<typename __VT>
struct __VectorMinReduce {
  static MLLM_CPU_ARM_FORCE_INLINE typename __VT::type cal(__VT v) {}
};

template<typename __VT>
struct __VectorMulReduce {
  static MLLM_CPU_ARM_FORCE_INLINE typename __VT::type cal(__VT v) {}
};

//===----------------------------------------------------------------------===//
// Int32 Vector Reduce Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorMaxReduce<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32_t cal(int32x4_t v) {
    int32x2_t max_val = vmax_s32(vget_high_s32(v), vget_low_s32(v));
    return vmaxv_s32(max_val);
  }
};

template<>
struct __VectorSumReduce<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32_t cal(int32x4_t v) {
    int32x2_t sum_val = vadd_s32(vget_high_s32(v), vget_low_s32(v));
    return vaddv_s32(sum_val);
  }
};

template<>
struct __VectorMinReduce<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32_t cal(int32x4_t v) {
    int32x2_t min_val = vmin_s32(vget_high_s32(v), vget_low_s32(v));
    return vminv_s32(min_val);
  }
};

template<>
struct __VectorMulReduce<int32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int32_t cal(int32x4_t v) {
    int32x2_t mul_val = vmul_s32(vget_high_s32(v), vget_low_s32(v));
    mul_val = vmul_s32(mul_val, vrev64_s32(mul_val));
    return vget_lane_s32(mul_val, 0);
  }
};

//===----------------------------------------------------------------------===//
// Int16 Vector Reduce Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorMaxReduce<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16_t cal(int16x8_t v) { return vmaxvq_s16(v); }
};

template<>
struct __VectorSumReduce<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16_t cal(int16x8_t v) { return vaddvq_s16(v); }
};

template<>
struct __VectorMinReduce<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16_t cal(int16x8_t v) { return vminvq_s16(v); }
};

template<>
struct __VectorMulReduce<int16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int16_t cal(int16x8_t v) {
    int16x4_t low = vget_low_s16(v);
    int16x4_t high = vget_high_s16(v);
    int16x4_t mul1 = vmul_s16(low, high);
    int16x4_t mul2 = vmul_s16(mul1, vrev64_s16(mul1));
    int16x4_t mul3 = vpadd_s16(mul2, mul2);
    return vget_lane_s16(mul3, 0) * vget_lane_s16(mul3, 1);
  }
};

//===----------------------------------------------------------------------===//
// Int8 Vector Reduce Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorMaxReduce<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8_t cal(int8x16_t v) { return vmaxvq_s8(v); }
};

template<>
struct __VectorSumReduce<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8_t cal(int8x16_t v) { return vaddvq_s8(v); }
};

template<>
struct __VectorMinReduce<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8_t cal(int8x16_t v) { return vminvq_s8(v); }
};

template<>
struct __VectorMulReduce<int8x16_t> {
  static MLLM_CPU_ARM_FORCE_INLINE int8_t cal(int8x16_t v) {
    int8x8_t low = vget_low_s8(v);
    int8x8_t high = vget_high_s8(v);
    int8x8_t mul1 = vmul_s8(low, high);
    int8x8_t mul2 = vmul_s8(mul1, vrev64_s8(mul1));
    int8x8_t mul3 = vpadd_s8(mul2, mul2);
    int8x8_t mul4 = vpadd_s8(mul3, mul3);
    return vget_lane_s8(mul4, 0) * vget_lane_s8(mul4, 1) * vget_lane_s8(mul4, 2) * vget_lane_s8(mul4, 3);
  }
};

//===----------------------------------------------------------------------===//
// Float32 Vector Reduce Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorMaxReduce<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float cal(float32x4_t v) { return vmaxvq_f32(v); }
};

template<>
struct __VectorSumReduce<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float cal(float32x4_t v) { return vaddvq_f32(v); }
};

template<>
struct __VectorMinReduce<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float cal(float32x4_t v) { return vminvq_f32(v); }
};

template<>
struct __VectorMulReduce<float32x4_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float cal(float32x4_t v) {
    float32x2_t low = vget_low_f32(v);
    float32x2_t high = vget_high_f32(v);
    float32x2_t mul = vmul_f32(low, high);
    mul = vmul_f32(mul, vrev64_f32(mul));
    return vget_lane_f32(mul, 0);
  }
};

//===----------------------------------------------------------------------===//
// Float16 Vector Reduce Ops
//===----------------------------------------------------------------------===//
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct __VectorMaxReduce<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16_t cal(float16x8_t v) { return vmaxvq_f16(v); }
};

template<>
struct __VectorSumReduce<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16_t cal(float16x8_t v) {
    float16x4_t low = vget_low_f16(v);
    float16x4_t high = vget_high_f16(v);
    float16x4_t sum1 = vadd_f16(low, high);
    float16x4_t sum2 = vpadd_f16(sum1, sum1);
    float16x4_t sum3 = vpadd_f16(sum2, sum2);
    return vget_lane_f16(sum3, 0);
  }
};

template<>
struct __VectorMinReduce<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16_t cal(float16x8_t v) { return vminvq_f16(v); }
};

template<>
struct __VectorMulReduce<float16x8_t> {
  static MLLM_CPU_ARM_FORCE_INLINE float16_t cal(float16x8_t v) {
    float16x4_t low = vget_low_f16(v);
    float16x4_t high = vget_high_f16(v);
    float16x4_t mul = vmul_f16(low, high);
    mul = vmul_f16(mul, vrev64_f16(mul));
    return vget_lane_f16(mul, 0);
  }
};
#endif

//===----------------------------------------------------------------------===//
// Scalar/Vector CLIP Ops
//===----------------------------------------------------------------------===//

template<typename __T>
struct __ScalarClip {
  __T min_val;
  __T max_val;

  __ScalarClip(__T _min_val, __T _max_val) : min_val(_min_val), max_val(_max_val) {}

  MLLM_CPU_ARM_FORCE_INLINE __T cal(__T a) { return (a < min_val) ? min_val : ((a > max_val) ? max_val : a); }
};

template<typename __VT>
struct __VectorClip {
  typename __VT::type min_val;
  typename __VT::type max_val;

  __VectorClip(typename __VT::type _min_val, typename __VT::type _max_val) : min_val(_min_val), max_val(_max_val) {}

  MLLM_CPU_ARM_FORCE_INLINE __VT cal(__VT a) {}

  MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 1; }
};

template<>
struct __VectorClip<int32x4_t> {
  int32_t min_val;
  int32_t max_val;

  __VectorClip(int32_t _min_val, int32_t _max_val) : min_val(_min_val), max_val(_max_val) {}

  MLLM_CPU_ARM_FORCE_INLINE int32x4_t cal(int32x4_t a) {
    int32x4_t min_vec = vdupq_n_s32(min_val);
    int32x4_t max_vec = vdupq_n_s32(max_val);
    return vminq_s32(vmaxq_s32(a, min_vec), max_vec);
  }

  MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorClip<int16x8_t> {
  int16_t min_val;
  int16_t max_val;

  __VectorClip(int16_t _min_val, int16_t _max_val) : min_val(_min_val), max_val(_max_val) {}

  MLLM_CPU_ARM_FORCE_INLINE int16x8_t cal(int16x8_t a) {
    int16x8_t min_vec = vdupq_n_s16(min_val);
    int16x8_t max_vec = vdupq_n_s16(max_val);
    return vminq_s16(vmaxq_s16(a, min_vec), max_vec);
  }

  MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorClip<int8x16_t> {
  int8_t min_val;
  int8_t max_val;

  __VectorClip(int8_t _min_val, int8_t _max_val) : min_val(_min_val), max_val(_max_val) {}

  MLLM_CPU_ARM_FORCE_INLINE int8x16_t cal(int8x16_t a) {
    int8x16_t min_vec = vdupq_n_s8(min_val);
    int8x16_t max_vec = vdupq_n_s8(max_val);
    return vminq_s8(vmaxq_s8(a, min_vec), max_vec);
  }

  MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorClip<float32x4_t> {
  float min_val;
  float max_val;

  __VectorClip(float _min_val, float _max_val) : min_val(_min_val), max_val(_max_val) {}

  MLLM_CPU_ARM_FORCE_INLINE float32x4_t cal(float32x4_t a) {
    float32x4_t min_vec = vdupq_n_f32(min_val);
    float32x4_t max_vec = vdupq_n_f32(max_val);
    return vminq_f32(vmaxq_f32(a, min_vec), max_vec);
  }

  MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 4; }
};

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct __VectorClip<float16x8_t> {
  float16_t min_val;
  float16_t max_val;

  __VectorClip(float16_t _min_val, float16_t _max_val) : min_val(_min_val), max_val(_max_val) {}

  MLLM_CPU_ARM_FORCE_INLINE float16x8_t cal(float16x8_t a) {
    float16x8_t min_vec = vdupq_n_f16(min_val);
    float16x8_t max_vec = vdupq_n_f16(max_val);
    return vminq_f16(vmaxq_f16(a, min_vec), max_vec);
  }

  MLLM_CPU_ARM_FORCE_INLINE constexpr size_t lanes() { return 8; }
};
#endif

}  // namespace mllm::cpu::arm

#endif
