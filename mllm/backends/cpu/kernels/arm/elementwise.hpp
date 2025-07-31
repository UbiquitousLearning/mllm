/**
 * @file elementwise.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 */
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

template<typename __T>
struct __ScalarAdd {
  static inline __T cal(__T a, __T b) { return a + b; }
};

template<typename __T>
struct __ScalarSub {
  static inline __T cal(__T a, __T b) { return a - b; }
};

template<typename __T>
struct __ScalarMul {
  static inline __T cal(__T a, __T b) { return a * b; }
};

template<typename __T>
struct __ScalarDiv {
  static inline __T cal(__T a, __T b) { return a / b; }
};

template<typename __VT>
struct __VectorAdd {
  static inline __VT cal(__VT a, __VT b) {}

  static inline constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorSub {
  static inline __VT cal(__VT a, __VT b) {}

  static inline constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorMul {
  static inline __VT cal(__VT a, __VT b) {}

  static inline constexpr size_t lanes() { return 1; }
};

template<typename __VT>
struct __VectorDiv {
  static inline __VT cal(__VT a, __VT b) {}

  static inline constexpr size_t lanes() { return 1; }
};

//===----------------------------------------------------------------------===//
// Int32 Vector Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorAdd<int32x4_t> {
  static inline int32x4_t cal(int32x4_t a, int32x4_t b) { return vaddq_s32(a, b); }

  static inline constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorSub<int32x4_t> {
  static inline int32x4_t cal(int32x4_t a, int32x4_t b) { return vsubq_s32(a, b); }

  static inline constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorMul<int32x4_t> {
  static inline int32x4_t cal(int32x4_t a, int32x4_t b) { return vmulq_s32(a, b); }

  static inline constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorDiv<int32x4_t> {
  static inline int32x4_t cal(int32x4_t a, int32x4_t b) {
    // Note: NEON does not have integer division, convert to float, divide, then convert back
    float32x4_t fa = vcvtq_f32_s32(a);
    float32x4_t fb = vcvtq_f32_s32(b);
    float32x4_t result = vdivq_f32(fa, fb);
    return vcvtq_s32_f32(result);
  }

  static inline constexpr size_t lanes() { return 4; }
};

//===----------------------------------------------------------------------===//
// Int16 Vector Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorAdd<int16x8_t> {
  static inline int16x8_t cal(int16x8_t a, int16x8_t b) { return vaddq_s16(a, b); }

  static inline constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorSub<int16x8_t> {
  static inline int16x8_t cal(int16x8_t a, int16x8_t b) { return vsubq_s16(a, b); }

  static inline constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorMul<int16x8_t> {
  static inline int16x8_t cal(int16x8_t a, int16x8_t b) { return vmulq_s16(a, b); }

  static inline constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorDiv<int16x8_t> {
  static inline int16x8_t cal(int16x8_t a, int16x8_t b) {
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

  static inline constexpr size_t lanes() { return 8; }
};

//===----------------------------------------------------------------------===//
// Int8 Vector Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorAdd<int8x16_t> {
  static inline int8x16_t cal(int8x16_t a, int8x16_t b) { return vaddq_s8(a, b); }

  static inline constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorSub<int8x16_t> {
  static inline int8x16_t cal(int8x16_t a, int8x16_t b) { return vsubq_s8(a, b); }
  static inline constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorMul<int8x16_t> {
  static inline int8x16_t cal(int8x16_t a, int8x16_t b) { return vmulq_s8(a, b); }

  static inline constexpr size_t lanes() { return 16; }
};

template<>
struct __VectorDiv<int8x16_t> {
  static inline int8x16_t cal(int8x16_t a, int8x16_t b) {
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

  static inline constexpr size_t lanes() { return 16; }
};

//===----------------------------------------------------------------------===//
// Float32 Vector Ops
//===----------------------------------------------------------------------===//
template<>
struct __VectorAdd<float32x4_t> {
  static inline float32x4_t cal(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }

  static inline constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorSub<float32x4_t> {
  static inline float32x4_t cal(float32x4_t a, float32x4_t b) { return vsubq_f32(a, b); }

  static inline constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorMul<float32x4_t> {
  static inline float32x4_t cal(float32x4_t a, float32x4_t b) { return vmulq_f32(a, b); }

  static inline constexpr size_t lanes() { return 4; }
};

template<>
struct __VectorDiv<float32x4_t> {
  static inline float32x4_t cal(float32x4_t a, float32x4_t b) { return vdivq_f32(a, b); }

  static inline constexpr size_t lanes() { return 4; }
};

//===----------------------------------------------------------------------===//
// Float16 Vector Ops
//===----------------------------------------------------------------------===//
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct __VectorAdd<float16x8_t> {
  static inline float16x8_t cal(float16x8_t a, float16x8_t b) { return vaddq_f16(a, b); }

  static inline constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorSub<float16x8_t> {
  static inline float16x8_t cal(float16x8_t a, float16x8_t b) { return vsubq_f16(a, b); }

  static inline constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorMul<float16x8_t> {
  static inline float16x8_t cal(float16x8_t a, float16x8_t b) { return vmulq_f16(a, b); }

  static inline constexpr size_t lanes() { return 8; }
};

template<>
struct __VectorDiv<float16x8_t> {
  static inline float16x8_t cal(float16x8_t a, float16x8_t b) { return vdivq_f16(a, b); }

  static inline constexpr size_t lanes() { return 8; }
};
#endif

//===----------------------------------------------------------------------===//
// Scalar Load Ops
//===----------------------------------------------------------------------===//
template<typename T>
struct __ScalarLoad {
  static inline T load(const T* __restrict__ src, size_t pos) { return *(src + pos); }
};

//===----------------------------------------------------------------------===//
// Scalar Store Ops
//===----------------------------------------------------------------------===//
template<typename T>
struct __ScalarStore {
  static inline void store(T* __restrict__ dst, size_t pos, T value) { *(dst + pos) = value; }
};

//===----------------------------------------------------------------------===//
// Vector Load Ops
//===----------------------------------------------------------------------===//
template<typename T>
struct __VectorLoad {
  static inline T load(const typename T::type* __restrict__ src, size_t pos) {}
};

template<>
struct __VectorLoad<int32x4_t> {
  static inline int32x4_t load(const int32_t* __restrict__ src, size_t pos) { return vld1q_s32(src + pos); }
  static inline int32x4_t load_scalar(const int32_t src) { return vdupq_n_s32(src); }
};

template<>
struct __VectorLoad<int16x8_t> {
  static inline int16x8_t load(const int16_t* __restrict__ src, size_t pos) { return vld1q_s16(src + pos); }
  static inline int16x8_t load_scalar(const int16_t src) { return vdupq_n_s16(src); }
};

template<>
struct __VectorLoad<int8x16_t> {
  static inline int8x16_t load(const int8_t* __restrict__ src, size_t pos) { return vld1q_s8(src + pos); }
  static inline int8x16_t load_scalar(const int8_t src) { return vdupq_n_s8(src); }
};

template<>
struct __VectorLoad<float32x4_t> {
  static inline float32x4_t load(const float* __restrict__ src, size_t pos) { return vld1q_f32(src + pos); }
  static inline float32x4_t load_scalar(const float src) { return vdupq_n_f32(src); }
};

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct __VectorLoad<float16x8_t> {
  static inline float16x8_t load(const float16_t* __restrict__ src, size_t pos) { return vld1q_f16(src + pos); }
  static inline float16x8_t load_scalar(const float16_t src) { return vdupq_n_f16(src); }
};
#endif

//===----------------------------------------------------------------------===//
// Vector Store Ops
//===----------------------------------------------------------------------===//
template<typename T>
struct __VectorStore {
  static inline void store(typename T::type* __restrict__ dst, size_t pos, T value) {}
};

template<>
struct __VectorStore<int32x4_t> {
  static inline void store(int32_t* __restrict__ dst, size_t pos, int32x4_t value) { vst1q_s32(dst + pos, value); }
};

template<>
struct __VectorStore<int16x8_t> {
  static inline void store(int16_t* __restrict__ dst, size_t pos, int16x8_t value) { vst1q_s16(dst + pos, value); }
};

template<>
struct __VectorStore<int8x16_t> {
  static inline void store(int8_t* __restrict__ dst, size_t pos, int8x16_t value) { vst1q_s8(dst + pos, value); }
};

template<>
struct __VectorStore<float32x4_t> {
  static inline void store(float* __restrict__ dst, size_t pos, float32x4_t value) { vst1q_f32(dst + pos, value); }
};

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct __VectorStore<float16x8_t> {
  static inline void store(float16_t* __restrict__ dst, size_t pos, float16x8_t value) { vst1q_f16(dst + pos, value); }
};
#endif

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//
// Element wise loop
template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct __ew_loop {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, const __ST* __restrict src1, size_t size) {
    size_t constexpr lanes = __VectorOp::lanes();
    size_t vec_size = size & ~(lanes - 1);
    size_t i = 0;
    for (; i + lanes * loop_unroll_size <= vec_size; i += lanes * loop_unroll_size) {
#pragma unroll
      for (size_t u = 0; u < loop_unroll_size; ++u) {
        __VT va = __VectorLoad<__VT>::load(src0, i + u * lanes);
        __VT vb = __VectorLoad<__VT>::load(src1, i + u * lanes);
        __VT result = __VectorOp::cal(va, vb);
        __VectorStore<__VT>::store(dst, i + u * lanes, result);
      }
    }
    for (; i < vec_size; i += lanes) {
      __VT va = __VectorLoad<__VT>::load(src0, i);
      __VT vb = __VectorLoad<__VT>::load(src1, i);
      __VT result = __VectorOp::cal(va, vb);
      __VectorStore<__VT>::store(dst, i, result);
    }
    for (; i < size; ++i) {
      __ST sa = __ScalarLoad<__ST>::load(src0, i);
      __ST sb = __ScalarLoad<__ST>::load(src1, i);
      __ST result = __ScalarOp::cal(sa, sb);
      __ScalarStore<__ST>::store(dst, i, result);
    }
  }
};

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct ParallelElementwiseLoop {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, const __ST* __restrict src1, size_t size,
                         size_t thread_count) {
    if (thread_count > 1) {
      size_t constexpr lanes = __VectorOp::lanes();
      size_t vec_size = size & ~(lanes - 1);
      size_t chunk_size = (vec_size + thread_count - 1) / thread_count;
      chunk_size = (chunk_size + lanes - 1) & ~(lanes - 1);

#pragma omp parallel for if (thread_count > 1) num_threads(thread_count)
      for (size_t start = 0; start < vec_size; start += chunk_size) {
        size_t end = std::min(start + chunk_size, vec_size);
        size_t local_size = end - start;
        __ew_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + start, src0 + start, src1 + start,
                                                                             local_size);
      }
      if (vec_size < size) {
        size_t remaining_size = size - vec_size;
        __ew_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + vec_size, src0 + vec_size, src1 + vec_size,
                                                                             remaining_size);
      }
    } else {
      __ew_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst, src0, src1, size);
    }
  }
};

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct __ew_loop_array_scalar {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, const __ST src1, size_t size) {
    size_t constexpr lanes = __VectorOp::lanes();
    size_t vec_size = size & ~(lanes - 1);
    size_t i = 0;

    __VT vb_scalar = __VectorLoad<__VT>::load_scalar(src1);

    for (; i + lanes * loop_unroll_size <= vec_size; i += lanes * loop_unroll_size) {
#pragma unroll
      for (size_t u = 0; u < loop_unroll_size; ++u) {
        __VT va = __VectorLoad<__VT>::load(src0, i + u * lanes);
        __VT result = __VectorOp::cal(va, vb_scalar);
        __VectorStore<__VT>::store(dst, i + u * lanes, result);
      }
    }
    for (; i < vec_size; i += lanes) {
      __VT va = __VectorLoad<__VT>::load(src0, i);
      __VT result = __VectorOp::cal(va, vb_scalar);
      __VectorStore<__VT>::store(dst, i, result);
    }
    for (; i < size; ++i) {
      __ST sa = __ScalarLoad<__ST>::load(src0, i);
      __ST result = __ScalarOp::cal(sa, src1);
      __ScalarStore<__ST>::store(dst, i, result);
    }
  }
};

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct ParallelElementwiseLoopArrayScalar {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, const __ST src1, size_t size, size_t thread_count) {
    if (thread_count > 1) {
      size_t constexpr lanes = __VectorOp::lanes();
      size_t vec_size = size & ~(lanes - 1);
      size_t chunk_size = (vec_size + thread_count - 1) / thread_count;
      chunk_size = (chunk_size + lanes - 1) & ~(lanes - 1);

#pragma omp parallel for if (thread_count > 1) num_threads(thread_count)
      for (size_t start = 0; start < vec_size; start += chunk_size) {
        size_t end = std::min(start + chunk_size, vec_size);
        size_t local_size = end - start;
        __ew_loop_array_scalar<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + start, src0 + start, src1,
                                                                                          local_size);
      }
      if (vec_size < size) {
        size_t remaining_size = size - vec_size;
        __ew_loop_array_scalar<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + vec_size, src0 + vec_size, src1,
                                                                                          remaining_size);
      }
    } else {
      __ew_loop_array_scalar<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst, src0, src1, size);
    }
  }
};

//===----------------------------------------------------------------------===//
// INSTANCE
//===----------------------------------------------------------------------===//
void ew_add_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_sub_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_mul_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_div_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_add_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_sub_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_mul_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_div_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count);
#endif

void ew_add_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_sub_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_mul_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_div_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_add_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_sub_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_mul_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_div_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_add_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_sub_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_mul_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_div_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_add_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count);

void ew_sub_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count);

void ew_mul_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count);

void ew_div_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_add_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count);

void ew_sub_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count);

void ew_mul_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count);

void ew_div_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count);
#endif

void ew_add_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count);

void ew_sub_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count);

void ew_mul_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count);

void ew_div_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count);

void ew_add_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count);

void ew_sub_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count);

void ew_mul_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count);

void ew_div_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count);

void ew_add_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count);

void ew_sub_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count);

void ew_mul_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count);

void ew_div_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count);
}  // namespace mllm::cpu::arm

#endif
