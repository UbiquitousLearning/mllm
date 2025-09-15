/*
 * This code is based on mllm(https://github.com/ggerganov/mllm),
 * please see https://github.com/ggerganov/mllm/blob/master/src/mllm.c
 * mllm is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
 *
 * MIT License
 * Copyright (c) 2022 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef MLLM_VECDOT_HPP
#define MLLM_VECDOT_HPP
#include "mllm/core/DataTypes.hpp"

#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
#include <sys/prctl.h>
#endif

namespace mllm::cpu::ggml {
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

// F32 NEON

#define MLLM_F32_STEP 16
#define MLLM_F32_EPR 4
#define MLLM_F32_ARR (MLLM_F32_STEP / MLLM_F32_EPR)
#define MLLM_F16_ARR (MLLM_F16_STEP / MLLM_F16_EPR)

#define MLLM_F32x4 float32x4_t
#define MLLM_F32x4_ZERO vdupq_n_f32(0.0f)
#define MLLM_F32x4_SET1(x) vdupq_n_f32(x)
#define MLLM_F32x4_LOAD vld1q_f32
#define MLLM_F32x4_STORE vst1q_f32
#define MLLM_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define MLLM_F32x4_ADD vaddq_f32
#define MLLM_F32x4_MUL vmulq_f32
#define MLLM_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#define MLLM_F32x4_REDUCE(res, x)                                               \
  {                                                                             \
    int offset = MLLM_F32_ARR >> 1;                                             \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f32(x[i], x[offset + i]); } \
    offset >>= 1;                                                               \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f32(x[i], x[offset + i]); } \
    offset >>= 1;                                                               \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f32(x[i], x[offset + i]); } \
    res = MLLM_F32x4_REDUCE_ONE(x[0]);                                          \
  }

#define MLLM_F32_VEC MLLM_F32x4
#define MLLM_F32_VEC_ZERO MLLM_F32x4_ZERO
#define MLLM_F32_VEC_SET1 MLLM_F32x4_SET1
#define MLLM_F32_VEC_LOAD MLLM_F32x4_LOAD
#define MLLM_F32_VEC_STORE MLLM_F32x4_STORE
#define MLLM_F32_VEC_FMA MLLM_F32x4_FMA
#define MLLM_F32_VEC_ADD MLLM_F32x4_ADD
#define MLLM_F32_VEC_MUL MLLM_F32x4_MUL
#define MLLM_F32_VEC_REDUCE MLLM_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define MLLM_F16_STEP 32
#define MLLM_F16_EPR 8
#define MLLM_F32_ARR (MLLM_F32_STEP / MLLM_F32_EPR)
#define MLLM_F16_ARR (MLLM_F16_STEP / MLLM_F16_EPR)

#define MLLM_F16x8 float16x8_t
#define MLLM_F16x8_ZERO vdupq_n_f16(0.0f)
#define MLLM_F16x8_SET1(x) vdupq_n_f16(x)
#define MLLM_F16x8_LOAD vld1q_f16
#define MLLM_F16x8_STORE vst1q_f16
#define MLLM_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
#define MLLM_F16x8_ADD vaddq_f16
#define MLLM_F16x8_MUL vmulq_f16
#define MLLM_F16x8_REDUCE(res, x)                                               \
  {                                                                             \
    int offset = MLLM_F16_ARR >> 1;                                             \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f16(x[i], x[offset + i]); } \
    offset >>= 1;                                                               \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f16(x[i], x[offset + i]); } \
    offset >>= 1;                                                               \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f16(x[i], x[offset + i]); } \
    const float32x4_t t0 = vcvt_f32_f16(vget_low_f16(x[0]));                    \
    const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(x[0]));                   \
    res = (float)vaddvq_f32(vaddq_f32(t0, t1));                                 \
  }

#define MLLM_F16_VEC MLLM_F16x8
#define MLLM_F16_VEC_ZERO MLLM_F16x8_ZERO
#define MLLM_F16_VEC_SET1 MLLM_F16x8_SET1
#define MLLM_F16_VEC_LOAD(p, i) MLLM_F16x8_LOAD(p)
#define MLLM_F16_VEC_STORE(p, r, i) MLLM_F16x8_STORE(p, r[i])
#define MLLM_F16_VEC_FMA MLLM_F16x8_FMA
#define MLLM_F16_VEC_ADD MLLM_F16x8_ADD
#define MLLM_F16_VEC_MUL MLLM_F16x8_MUL
#define MLLM_F16_VEC_REDUCE MLLM_F16x8_REDUCE
#else
// if FP16 vector arithmetic is not supported, we use FP32 instead
// and take advantage of the vcvt_ functions to convert to/from FP16

#define MLLM_F16_STEP 16
#define MLLM_F16_EPR 4

#define MLLM_F32Cx4 float32x4_t
#define MLLM_F32Cx4_ZERO vdupq_n_f32(0.0f)
#define MLLM_F32Cx4_SET1(x) vdupq_n_f32(x)
#define MLLM_F32Cx4_LOAD(x) vcvt_f32_f16(vld1_f16(x))
#define MLLM_F32Cx4_STORE(x, y) vst1_f16(x, vcvt_f16_f32(y))
#define MLLM_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define MLLM_F32Cx4_ADD vaddq_f32
#define MLLM_F32Cx4_MUL vmulq_f32
#define MLLM_F32Cx4_REDUCE MLLM_F32x4_REDUCE

#define MLLM_F16_VEC MLLM_F32Cx4
#define MLLM_F16_VEC_ZERO MLLM_F32Cx4_ZERO
#define MLLM_F16_VEC_SET1 MLLM_F32Cx4_SET1
#define MLLM_F16_VEC_LOAD(p, i) MLLM_F32Cx4_LOAD(p)
#define MLLM_F16_VEC_STORE(p, r, i) MLLM_F32Cx4_STORE(p, r[i])
#define MLLM_F16_VEC_FMA MLLM_F32Cx4_FMA
#define MLLM_F16_VEC_ADD MLLM_F32Cx4_ADD
#define MLLM_F16_VEC_MUL MLLM_F32Cx4_MUL
#define MLLM_F16_VEC_REDUCE MLLM_F32Cx4_REDUCE
#endif

#elif __AVX2__
//  COPY FROM MLLM
#define MLLM_F32_STEP 32
#define MLLM_F32_EPR 8
#define MLLM_F32_ARR (MLLM_F32_STEP / MLLM_F32_EPR)
#define MLLM_F16_ARR (MLLM_F16_STEP / MLLM_F16_EPR)
#define MLLM_F32x8 __m256
#define MLLM_F32x8_ZERO _mm256_setzero_ps()
#define MLLM_F32x8_SET1(x) _mm256_set1_ps(x)
#define MLLM_F32x8_LOAD _mm256_loadu_ps
#define MLLM_F32x8_STORE _mm256_storeu_ps
#if defined(__FMA__)
#define MLLM_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
#define MLLM_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define MLLM_F32x8_ADD _mm256_add_ps
#define MLLM_F32x8_MUL _mm256_mul_ps
#define MLLM_F32x8_REDUCE(res, x)                                                               \
  {                                                                                             \
    int offset = MLLM_F32_ARR >> 1;                                                             \
    for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset + i]); }             \
    offset >>= 1;                                                                               \
    for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset + i]); }             \
    offset >>= 1;                                                                               \
    for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset + i]); }             \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                                                      \
    res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));                                                   \
  }
#define MLLM_F32x8 __m256
#define MLLM_F32_VEC MLLM_F32x8
#define MLLM_F32_VEC_ZERO MLLM_F32x8_ZERO
#define MLLM_F32_VEC_SET1 MLLM_F32x8_SET1
#define MLLM_F32_VEC_LOAD MLLM_F32x8_LOAD
#define MLLM_F32_VEC_STORE MLLM_F32x8_STORE
#define MLLM_F32_VEC_FMA MLLM_F32x8_FMA
#define MLLM_F32_VEC_ADD MLLM_F32x8_ADD
#define MLLM_F32_VEC_MUL MLLM_F32x8_MUL
#define MLLM_F32_VEC_REDUCE MLLM_F32x8_REDUCE
// F16 AVX

#define MLLM_F16_STEP 32
#define MLLM_F16_EPR 8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define MLLM_F32Cx8 __m256
#define MLLM_F32Cx8_ZERO _mm256_setzero_ps()
#define MLLM_F32Cx8_SET1(x) _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define MLLM_F32Cx8_LOAD(x) _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x)))
#define MLLM_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i*)(x), _mm256_cvtps_ph(y, 0))
#else
static inline __m256 __avx_f32cx8_load(MLLM_fp16_t* x) {
  float tmp[8];

  for (int i = 0; i < 8; i++) { tmp[i] = MLLM_FP16_TO_FP32(x[i]); }

  return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(MLLM_fp16_t* x, __m256 y) {
  float arr[8];

  _mm256_storeu_ps(arr, y);

  for (int i = 0; i < 8; i++) x[i] = MLLM_FP32_TO_FP16(arr[i]);
}
#define MLLM_F32Cx8_LOAD(x) __avx_f32cx8_load(x)
#define MLLM_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#endif

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#define MLLM_F32Cx8_FMA MLLM_F32x8_FMA
#define MLLM_F32Cx8_ADD _mm256_add_ps
#define MLLM_F32Cx8_MUL _mm256_mul_ps
#define MLLM_F32Cx8_REDUCE MLLM_F32x8_REDUCE

#define MLLM_F16_VEC MLLM_F32Cx8
#define MLLM_F16_VEC_ZERO MLLM_F32Cx8_ZERO
#define MLLM_F16_VEC_SET1 MLLM_F32Cx8_SET1
#define MLLM_F16_VEC_LOAD(p, i) MLLM_F32Cx8_LOAD(p)
#define MLLM_F16_VEC_STORE(p, r, i) MLLM_F32Cx8_STORE(p, r[i])
#define MLLM_F16_VEC_FMA MLLM_F32Cx8_FMA
#define MLLM_F16_VEC_ADD MLLM_F32Cx8_ADD
#define MLLM_F16_VEC_MUL MLLM_F32Cx8_MUL
#define MLLM_F16_VEC_REDUCE MLLM_F32Cx8_REDUCE

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
  const __m128i tmp = _mm_loadu_si128((const __m128i*)rsi);
  const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
  const __m256i lowMask = _mm256_set1_epi8(0xF);
  return _mm256_and_si256(lowMask, bytes);
}
// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
  return _mm256_cvtepi32_ps(summed_pairs);
}
static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVXVNNI__)
  const __m256i zero = _mm256_setzero_si256();
  // const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
  const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
  return _mm256_cvtepi32_ps(summed_pairs);
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
  // const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(ax, sy);
  return sum_i16_pairs_float(dot);
#endif
}
// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Get absolute values of x vectors
  const __m256i ax = _mm256_sign_epi8(x, x);
  // Sign the values of the y vectors
  const __m256i sy = _mm256_sign_epi8(y, x);
  return mul_sum_us8_pairs_float(ax, sy);
#endif
}

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}
#endif

#ifdef __ARM_NEON
#define COMPUTE_FP16_TO_FP32(x) ((float)(x))
#define COMPUTE_FP32_TO_FP16(x) (x)
#define FP16_TO_FP32(x) ((float)(x))
#define FP32_TO_FP16(x) (x)
#define F32_VEC float32x4_t
#define F32_STEP 16                 // 16 elements per step
#define F32_REG 4                   // 4 elements per register
#define F32_ARR F32_STEP / F32_REG  // Len of sum array
#define F32_VEC_REDUCE(res, x)                                                  \
  {                                                                             \
    int offset = F32_ARR >> 1;                                                  \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f32(x[i], x[offset + i]); } \
    offset >>= 1;                                                               \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f32(x[i], x[offset + i]); } \
    offset >>= 1;                                                               \
    for (int i = 0; i < offset; ++i) { x[i] = vaddq_f32(x[i], x[offset + i]); } \
    res = vaddvq_f32(x[0]);                                                     \
  }

#if !defined(__ARM_FEATURE_DOTPROD)

inline static int32x4_t mllm_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
  const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
  const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

  return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#else

#define mllm_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif  // !defined(__ARM_FEATURE_DOTPROD)

#endif

inline int mllm_cpu_get_sve_cnt() {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
  return PR_SVE_VL_LEN_MASK & prctl(PR_SVE_GET_VL);
#else
  return 0;
#endif
}

#if defined(__ARM_NEON)

// ref: https://github.com/mllm-org/llama.cpp/pull/5404
#ifdef _MSC_VER
#define mllm_vld1q_u32(w, x, y, z) {((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32))}
#else
#define mllm_vld1q_u32(w, x, y, z) {(w), (x), (y), (z)}
#endif  // _MSC_VER

#if !defined(__aarch64__)

// 32-bit ARM compatibility

// vaddlvq_s16
// vpaddq_s16
// vpaddq_s32
// vaddvq_s32
// vaddvq_f32
// vmaxvq_f32
// vcvtnq_s32_f32
// vzip1_u8
// vzip2_u8

inline static int32_t vaddlvq_s16(int16x8_t v) {
  int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
  return vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

inline static int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
  int16x4_t a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
  int16x4_t b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
  return vcombine_s16(a0, b0);
}

inline static int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
  int32x2_t a0 = vpadd_s32(vget_low_s32(a), vget_high_s32(a));
  int32x2_t b0 = vpadd_s32(vget_low_s32(b), vget_high_s32(b));
  return vcombine_s32(a0, b0);
}

inline static int32_t vaddvq_s32(int32x4_t v) {
  return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

inline static float vaddvq_f32(float32x4_t v) {
  return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

inline static float vmaxvq_f32(float32x4_t v) {
  return MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)), MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

inline static int32x4_t vcvtnq_s32_f32(float32x4_t v) {
  int32x4_t res;

  res[0] = roundf(vgetq_lane_f32(v, 0));
  res[1] = roundf(vgetq_lane_f32(v, 1));
  res[2] = roundf(vgetq_lane_f32(v, 2));
  res[3] = roundf(vgetq_lane_f32(v, 3));

  return res;
}

inline static uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
  uint8x8_t res;

  res[0] = a[0];
  res[1] = b[0];
  res[2] = a[1];
  res[3] = b[1];
  res[4] = a[2];
  res[5] = b[2];
  res[6] = a[3];
  res[7] = b[3];

  return res;
}

inline static uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
  uint8x8_t res;

  res[0] = a[4];
  res[1] = b[4];
  res[2] = a[5];
  res[3] = b[5];
  res[4] = a[6];
  res[5] = b[6];
  res[6] = a[7];
  res[7] = b[7];

  return res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

typedef struct mllm_int16x8x2_t {
  int16x8_t val[2];
} mllm_int16x8x2_t;

inline static mllm_int16x8x2_t mllm_vld1q_s16_x2(const int16_t* ptr) {
  mllm_int16x8x2_t res;

  res.val[0] = vld1q_s16(ptr + 0);
  res.val[1] = vld1q_s16(ptr + 8);

  return res;
}

typedef struct mllm_uint8x16x2_t {
  uint8x16_t val[2];
} mllm_uint8x16x2_t;

inline static mllm_uint8x16x2_t mllm_vld1q_u8_x2(const uint8_t* ptr) {
  mllm_uint8x16x2_t res;

  res.val[0] = vld1q_u8(ptr + 0);
  res.val[1] = vld1q_u8(ptr + 16);

  return res;
}

typedef struct mllm_uint8x16x4_t {
  uint8x16_t val[4];
} mllm_uint8x16x4_t;

inline static mllm_uint8x16x4_t mllm_vld1q_u8_x4(const uint8_t* ptr) {
  mllm_uint8x16x4_t res;

  res.val[0] = vld1q_u8(ptr + 0);
  res.val[1] = vld1q_u8(ptr + 16);
  res.val[2] = vld1q_u8(ptr + 32);
  res.val[3] = vld1q_u8(ptr + 48);

  return res;
}

typedef struct mllm_int8x16x2_t {
  int8x16_t val[2];
} mllm_int8x16x2_t;

inline static mllm_int8x16x2_t mllm_vld1q_s8_x2(const int8_t* ptr) {
  mllm_int8x16x2_t res;

  res.val[0] = vld1q_s8(ptr + 0);
  res.val[1] = vld1q_s8(ptr + 16);

  return res;
}

typedef struct mllm_int8x16x4_t {
  int8x16_t val[4];
} mllm_int8x16x4_t;

inline static mllm_int8x16x4_t mllm_vld1q_s8_x4(const int8_t* ptr) {
  mllm_int8x16x4_t res;

  res.val[0] = vld1q_s8(ptr + 0);
  res.val[1] = vld1q_s8(ptr + 16);
  res.val[2] = vld1q_s8(ptr + 32);
  res.val[3] = vld1q_s8(ptr + 48);

  return res;
}

// NOTE: not tested
inline static int8x16_t mllm_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
  int8x16_t res;

  res[0] = a[b[0]];
  res[1] = a[b[1]];
  res[2] = a[b[2]];
  res[3] = a[b[3]];
  res[4] = a[b[4]];
  res[5] = a[b[5]];
  res[6] = a[b[6]];
  res[7] = a[b[7]];
  res[8] = a[b[8]];
  res[9] = a[b[9]];
  res[10] = a[b[10]];
  res[11] = a[b[11]];
  res[12] = a[b[12]];
  res[13] = a[b[13]];
  res[14] = a[b[14]];
  res[15] = a[b[15]];

  return res;
}

// NOTE: not tested
inline static uint8x16_t mllm_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
  uint8x16_t res;

  res[0] = a[b[0]];
  res[1] = a[b[1]];
  res[2] = a[b[2]];
  res[3] = a[b[3]];
  res[4] = a[b[4]];
  res[5] = a[b[5]];
  res[6] = a[b[6]];
  res[7] = a[b[7]];
  res[8] = a[b[8]];
  res[9] = a[b[9]];
  res[10] = a[b[10]];
  res[11] = a[b[11]];
  res[12] = a[b[12]];
  res[13] = a[b[13]];
  res[14] = a[b[14]];
  res[15] = a[b[15]];

  return res;
}

#else

#define mllm_int16x8x2_t int16x8x2_t
#define mllm_uint8x16x2_t uint8x16x2_t
#define mllm_uint8x16x4_t uint8x16x4_t
#define mllm_int8x16x2_t int8x16x2_t
#define mllm_int8x16x4_t int8x16x4_t

#define mllm_vld1q_s16_x2 vld1q_s16_x2
#define mllm_vld1q_u8_x2 vld1q_u8_x2
#define mllm_vld1q_u8_x4 vld1q_u8_x4
#define mllm_vld1q_s8_x2 vld1q_s8_x2
#define mllm_vld1q_s8_x4 vld1q_s8_x4
#define mllm_vqtbl1q_s8 vqtbl1q_s8
#define mllm_vqtbl1q_u8 vqtbl1q_u8

#endif  // !defined(__aarch64__)
#endif  // !defined(__ARM_NEON)

inline static void vec_scale_f32(const int n, float* y, const float v) {
  const int np = (n & ~(MLLM_F32_STEP - 1));

  MLLM_F32_VEC vx = MLLM_F32_VEC_SET1(v);

  MLLM_F32_VEC ay[MLLM_F32_ARR];

  for (int i = 0; i < np; i += MLLM_F32_STEP) {
    for (int j = 0; j < MLLM_F32_ARR; j++) {
      ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);
      ay[j] = MLLM_F32_VEC_MUL(ay[j], vx);

      MLLM_F32_VEC_STORE(y + i + j * MLLM_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) { y[i] *= v; }

  //    for (int i = 0; i < n; ++i) {
  //        y[i] *= v;
  //    }
}

void vec_dot_q4_K_q8_K(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy);
void vec_dot_q6_K_q8_K(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy);
void vec_dot_q4_0_q8_0(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy);
void vec_dot_fp32(const int n, float* __restrict s, const float* __restrict vx, const float* __restrict vy);
void vec_dot_fp16(const int n, float* __restrict s, const mllm::mllm_fp16_t* __restrict vx,
                  const mllm::mllm_fp16_t* __restrict vy);
void vec_dot_q8_0_q8_0(int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy, size_t bs = 0,
                       size_t bx = 0, size_t by = 0);

void vec_dot_q2_K_q8_K(int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy);
void vec_dot_q3_K_q8_K(int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy);
void vec_dot_iq2_xxs_q8_K(int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy);

// for sparse linear
void vec_value_dot_fp32(const int n, float* __restrict s, const float x, const float* __restrict vy, bool addition);
// for per-tensor i8, currently not suitable for vecdot trait
void vec_dot_i8_i8(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy, float scale1 = 1,
                   float scale2 = 1);

}  // namespace mllm::cpu::ggml

#endif  // MLLM_VECDOT_HPP
