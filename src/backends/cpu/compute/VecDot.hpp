//
// Created by ey on 23-10-30.
//

#ifndef MLLM_VECDOT_HPP
#define MLLM_VECDOT_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include <functional>
#include "ParamLoader.hpp"
#include "../quantize/QuantizeQ8.hpp"
#include "../quantize/QuantizeQ4.hpp"


#include <chrono>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

// F32 NEON

#define MLLM_F32_STEP 16
#define MLLM_F32_EPR  4
#define MLLM_F32_ARR (MLLM_F32_STEP/MLLM_F32_EPR)
#define MLLM_F16_ARR (MLLM_F16_STEP/MLLM_F16_EPR)

#define MLLM_F32x4              float32x4_t
#define MLLM_F32x4_ZERO         vdupq_n_f32(0.0f)
#define MLLM_F32x4_SET1(x)      vdupq_n_f32(x)
#define MLLM_F32x4_LOAD         vld1q_f32
#define MLLM_F32x4_STORE        vst1q_f32
#define MLLM_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define MLLM_F32x4_ADD          vaddq_f32
#define MLLM_F32x4_MUL          vmulq_f32
#define MLLM_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#define MLLM_F32x4_REDUCE(res, x)              \
{                                              \
    int offset = MLLM_F32_ARR >> 1;            \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    res = MLLM_F32x4_REDUCE_ONE(x[0]);         \
}

#define MLLM_F32_VEC        MLLM_F32x4
#define MLLM_F32_VEC_ZERO   MLLM_F32x4_ZERO
#define MLLM_F32_VEC_SET1   MLLM_F32x4_SET1
#define MLLM_F32_VEC_LOAD   MLLM_F32x4_LOAD
#define MLLM_F32_VEC_STORE  MLLM_F32x4_STORE
#define MLLM_F32_VEC_FMA    MLLM_F32x4_FMA
#define MLLM_F32_VEC_ADD    MLLM_F32x4_ADD
#define MLLM_F32_VEC_MUL    MLLM_F32x4_MUL
#define MLLM_F32_VEC_REDUCE MLLM_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define MLLM_F16_STEP 32
#define MLLM_F16_EPR  8
#define MLLM_F32_ARR (MLLM_F32_STEP/MLLM_F32_EPR)
#define MLLM_F16_ARR (MLLM_F16_STEP/MLLM_F16_EPR)

#define MLLM_F16x8              float16x8_t
#define MLLM_F16x8_ZERO         vdupq_n_f16(0.0f)
#define MLLM_F16x8_SET1(x)      vdupq_n_f16(x)
#define MLLM_F16x8_LOAD         vld1q_f16
#define MLLM_F16x8_STORE        vst1q_f16
#define MLLM_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
#define MLLM_F16x8_ADD          vaddq_f16
#define MLLM_F16x8_MUL          vmulq_f16
#define MLLM_F16x8_REDUCE(res, x)                             \
    {                                                             \
        int offset = MLLM_F16_ARR >> 1;                           \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        offset >>= 1;                                             \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        offset >>= 1;                                             \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 (x[0])); \
        const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(x[0])); \
        res = (float) vaddvq_f32(vaddq_f32(t0, t1));         \
    }

#define MLLM_F16_VEC                MLLM_F16x8
#define MLLM_F16_VEC_ZERO           MLLM_F16x8_ZERO
#define MLLM_F16_VEC_SET1           MLLM_F16x8_SET1
#define MLLM_F16_VEC_LOAD(p, i)     MLLM_F16x8_LOAD(p)
#define MLLM_F16_VEC_STORE(p, r, i) MLLM_F16x8_STORE(p, r[i])
#define MLLM_F16_VEC_FMA            MLLM_F16x8_FMA
#define MLLM_F16_VEC_ADD            MLLM_F16x8_ADD
#define MLLM_F16_VEC_MUL            MLLM_F16x8_MUL
#define MLLM_F16_VEC_REDUCE         MLLM_F16x8_REDUCE
#else
// if FP16 vector arithmetic is not supported, we use FP32 instead
// and take advantage of the vcvt_ functions to convert to/from FP16

#define MLLM_F16_STEP 16
#define MLLM_F16_EPR  4

#define MLLM_F32Cx4              float32x4_t
#define MLLM_F32Cx4_ZERO         vdupq_n_f32(0.0f)
#define MLLM_F32Cx4_SET1(x)      vdupq_n_f32(x)
#define MLLM_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16(x))
#define MLLM_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
#define MLLM_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define MLLM_F32Cx4_ADD          vaddq_f32
#define MLLM_F32Cx4_MUL          vmulq_f32
#define MLLM_F32Cx4_REDUCE       MLLM_F32x4_REDUCE

#define MLLM_F16_VEC                MLLM_F32Cx4
#define MLLM_F16_VEC_ZERO           MLLM_F32Cx4_ZERO
#define MLLM_F16_VEC_SET1           MLLM_F32Cx4_SET1
#define MLLM_F16_VEC_LOAD(p, i)     MLLM_F32Cx4_LOAD(p)
#define MLLM_F16_VEC_STORE(p, r, i) MLLM_F32Cx4_STORE(p, r[i])
#define MLLM_F16_VEC_FMA            MLLM_F32Cx4_FMA
#define MLLM_F16_VEC_ADD            MLLM_F32Cx4_ADD
#define MLLM_F16_VEC_MUL            MLLM_F32Cx4_MUL
#define MLLM_F16_VEC_REDUCE         MLLM_F32Cx4_REDUCE
#endif

#elif  __AVX2__
//  COPY FROM MLLM
#define MLLM_F32_STEP 32
#define MLLM_F32_EPR 8
#define MLLM_F32_ARR (MLLM_F32_STEP/MLLM_F32_EPR)
#define MLLM_F16_ARR (MLLM_F16_STEP/MLLM_F16_EPR)
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
#define MLLM_F32x8_REDUCE(res, x)                                     \
    {                                                                 \
        int offset = MLLM_F32_ARR >> 1;                               \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset + i]);                \
        }                                                             \
        offset >>= 1;                                                 \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset + i]);                \
        }                                                             \
        offset >>= 1;                                                 \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset + i]);                \
        }                                                             \
        const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                     _mm256_extractf128_ps(x[0], 1)); \
        const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
        res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));                     \
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
#define MLLM_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define MLLM_F32Cx8             __m256
#define MLLM_F32Cx8_ZERO        _mm256_setzero_ps()
#define MLLM_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define MLLM_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x)))
#define MLLM_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#else
static inline __m256 __avx_f32cx8_load(MLLM_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = MLLM_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(MLLM_fp16_t *x, __m256 y) {
    float arr[8];

    _mm256_storeu_ps(arr, y);

    for (int i = 0; i < 8; i++)
        x[i] = MLLM_FP32_TO_FP16(arr[i]);
}
#define MLLM_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define MLLM_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#endif


#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#define MLLM_F32Cx8_FMA         MLLM_F32x8_FMA
#define MLLM_F32Cx8_ADD         _mm256_add_ps
#define MLLM_F32Cx8_MUL         _mm256_mul_ps
#define MLLM_F32Cx8_REDUCE      MLLM_F32x8_REDUCE

#define MLLM_F16_VEC                MLLM_F32Cx8
#define MLLM_F16_VEC_ZERO           MLLM_F32Cx8_ZERO
#define MLLM_F16_VEC_SET1           MLLM_F32Cx8_SET1
#define MLLM_F16_VEC_LOAD(p, i)     MLLM_F32Cx8_LOAD(p)
#define MLLM_F16_VEC_STORE(p, r, i) MLLM_F32Cx8_STORE(p, r[i])
#define MLLM_F16_VEC_FMA            MLLM_F32Cx8_FMA
#define MLLM_F16_VEC_ADD            MLLM_F32Cx8_ADD
#define MLLM_F16_VEC_MUL            MLLM_F32Cx8_MUL
#define MLLM_F16_VEC_REDUCE         MLLM_F32Cx8_REDUCE


// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}
// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}
static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if __AVXVNNI__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
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
#define COMPUTE_FP16_TO_FP32(x) ((float) (x))
#define COMPUTE_FP32_TO_FP16(x) (x)
#define FP16_TO_FP32(x) ((float) (x))
#define FP32_TO_FP16(x) (x)
#define F32_VEC float32x4_t
#define F32_STEP 16                // 16 elements per step
#define F32_REG 4                  // 4 elements per register
#define F32_ARR F32_STEP / F32_REG // Len of sum array
#define F32_VEC_REDUCE(res, x)                     \
    {                                              \
        int offset = F32_ARR >> 1;                 \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        offset >>= 1;                              \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        offset >>= 1;                              \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        res = vaddvq_f32(x[0]);                    \
    }
#endif

using namespace mllm;

void vec_dot_fp32(const float * __restrict src0, const float * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf);
void vec_dot_q4_0_q8_0(const void * __restrict src0, const void * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf);
void vec_dot_q4_K_q8_K(const void * __restrict src0, const void * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf);
void vec_dot_q6_K_q8_K(const void * __restrict src0, const void * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf);


void vec_dot_q4_K_q8_K(const int n, float * __restrict s, const void * __restrict vx, const void * __restrict vy);
void vec_dot_q6_K_q8_K(const int n, float * __restrict s, const void * __restrict vx, const void * __restrict vy);
void vec_dot_q4_0_q8_0(const int n, float * __restrict s, const void * __restrict vx, const void * __restrict vy);
void vec_dot_fp32(const int n, float * __restrict s, const float * __restrict vx, const float * __restrict vy);
void vec_dot_fp16(const int n, float * __restrict s, const mllm_fp16_t * __restrict vx, const mllm_fp16_t * __restrict vy);

#endif // MLLM_VECDOT_HPP
