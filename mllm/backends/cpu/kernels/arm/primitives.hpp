// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

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
#endif

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
    float16x4_t sum = vadd_f16(low, high);
    sum = vpadd_f16(sum, sum);
    sum = vpadd_f16(sum, sum);
    return vget_lane_f16(sum, 0);
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

}  // namespace mllm::cpu::arm

#endif
