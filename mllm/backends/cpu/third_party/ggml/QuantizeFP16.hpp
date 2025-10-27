/*
 * This code is based on ggml(https://github.com/ggerganov/ggml),
 * please see https://github.com/ggerganov/ggml/blob/master/src/ggml.c
 * ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
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

#ifndef MLLM_QUANTIZEFP16_HPP
#define MLLM_QUANTIZEFP16_HPP

#include "stdint.h"
#include "assert.h"
#include "math.h"
#include <string.h>
#include "Types.hpp"
#include <omp.h>

// #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
// #include <x86intrin.h>
// #endif

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
#ifdef __ARM_NEON

#else

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#else
#ifdef __POWER9_VECTOR__
#include <altivec.h>
#undef bool
#define bool _Bool
#else
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#if !defined(__riscv)
#include <immintrin.h>
#endif
#endif
#endif
#endif
#endif

// fp32<->fp16 start //
#if defined(__ARM_NEON) && !defined(_MSC_VER)
#include <arm_neon.h>
#define MLLM_COMPUTE_FP16_TO_FP32(x) ((float)(x))
#define MLLM_COMPUTE_FP32_TO_FP16(x) ((mllm_fp16_t)x)

#define MLLM_FP16_TO_FP32(x) ((float)(x))
#define MLLM_FP32_TO_FP16(x) ((mllm_fp16_t)x)

#elif defined _MSC_VER
#define MLLM_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define MLLM_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)

static float table_f32_f16[1 << 16];
static bool table_f32_f16_init = false;

inline static float lookup_fp16_to_fp32(uint16_t f) {
    if (!table_f32_f16_init) {
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
#define MLLM_FP32_TO_FP16(x) MLLM_COMPUTE_FP32_TO_FP16(x)

#else
#define MLLM_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define MLLM_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)

static float table_f32_f16[1 << 16];
static bool table_f32_f16_init = false;

inline static float lookup_fp16_to_fp32(uint16_t f) {
    if (!table_f32_f16_init) {
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
#define MLLM_FP32_TO_FP16(x) MLLM_COMPUTE_FP32_TO_FP16(x)
#endif

static mllm_fp16_t table_exp_f16[1 << 16];
static bool init_table_exp_f16_flag = false;
inline void init_table_exp_f16() {
    mllm_fp16_t ii;
    for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = MLLM_COMPUTE_FP16_TO_FP32(ii);
        table_exp_f16[i] = MLLM_FP32_TO_FP16(expf(f));
    }
}

// fp32<->fp16 end //

#endif // MLLM_QUANTIZEFP16_HPP