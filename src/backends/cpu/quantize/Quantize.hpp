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
#if defined(__ARM_NEON) && !defined(_MSC_VER)
#include <arm_neon.h>

#define MLLM_COMPUTE_FP16_TO_FP32(x) ((float) (x))
#define MLLM_COMPUTE_FP32_TO_FP16(x) (x)

#define MLLM_FP16_TO_FP32(x) ((float) (x))
#define MLLM_FP32_TO_FP16(x) (x)

#else
#ifdef _MSC_VER
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



#endif // MLLM_QUANTIZE_HPP
