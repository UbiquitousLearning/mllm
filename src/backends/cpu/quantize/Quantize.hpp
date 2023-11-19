//
// Created by ey on 23-10-24.
//

#ifndef MLLM_QUANTIZE_HPP
#define MLLM_QUANTIZE_HPP


#include "stdint.h"
#include "assert.h"
#include "math.h"
#include <string.h>
#include <iostream>
// TODO: better arch define macro
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif
#include "Types.hpp"
//#include <unistd.h>


#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


 // 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
#define mllm_fp16_t uint16_t

#if defined(__ARM_NEON) && !defined(_MSC_VER)
#include <arm_neon.h>
#define mllm_fp16_t __fp16
#define MLLM_COMPUTE_FP16_TO_FP32(x) ((float) (x))
#define MLLM_COMPUTE_FP32_TO_FP16(x) (x)

#define MLLM_FP16_TO_FP32(x) ((float) (x))
#define MLLM_FP32_TO_FP16(x) (x)

#elif defined _MSC_VER
#define MLLM_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define MLLM_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define MLLM_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define MLLM_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

static float table_f32_f16[1 << 16];
static bool table_f32_f16_init = false;

inline static float lookup_fp16_to_fp32(uint16_t f) {
    if(!table_f32_f16_init) {
        uint16_t ii;
        for (int i = 0; i < (1 << 16); ++i) {
            uint16_t ui = i;
            memcpy(&ii, &ui, sizeof(ii));
            table_f32_f16[i] = MLLM_COMPUTE_FP16_TO_FP32(ii);
        }
        table_f32_f16_init = true;
    }
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return table_f32_f16[s];
}

#define MLLM_FP16_TO_FP32(x) lookup_fp16_to_fp32(x)
#define MLLM_FP32_TO_FP16(x)  MLLM_COMPUTE_FP32_TO_FP16(x)
#endif


#if  __AVX2__
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t KShuffle[256] = {
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)KShuffle + i);
}
#endif

static inline int nearest_int(float fval) {
    assert(fval <= 4194303.F);
    float val = fval + 12582912.F;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

inline mllm_fp16_t mllm_fp32_to_fp16(float x) {
    return MLLM_FP32_TO_FP16(x);
}

inline void mllm_fp16_to_fp32_row(const mllm_fp16_t * x, float * y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = MLLM_FP16_TO_FP32(x[i]);
    }
}

inline void mllm_fp32_to_fp16_row(const float * x, mllm_fp16_t * y, int n) {
    int i = 0;
#if defined(__F16C__)
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for(; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#endif
    for (; i < n; i++) {
        y[i] = MLLM_FP32_TO_FP16(x[i]);
    }
}

#endif // MLLM_QUANTIZE_HPP
