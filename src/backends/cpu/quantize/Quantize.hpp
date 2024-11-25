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

#ifndef MLLM_QUANTIZE_HPP
#define MLLM_QUANTIZE_HPP

#include "stdint.h"
#include "assert.h"
#include "math.h"
#include <string.h>
#include <iostream>
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
        //        float val = MLLM_FP16_TO_FP32(expf(f));
        //        std::cout<<i<<"  "<<f<<" "<<expf(f)<<"  "<<val<<std::endl;
        //        printf("%d  %f %f  %f\n", i, f, expf(f), val);
    }
}
/*
inline double mllm_table_exp(float input){
    uint16_t scvt;
    mllm_fp16_t tmp = MLLM_FP32_TO_FP16(input);
    memcpy(&scvt, &tmp, sizeof(scvt));
    const float val = MLLM_FP16_TO_FP32(table_exp_f16[scvt]);
    return (double)val ;
}
*/

static const float GELU_COEF_A = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

inline static float mllm_gelu_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

inline static float mllm_gelu_quick_f32(float x) {
    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

// Sigmoid Linear Unit (SiLU) function
inline static float mllm_silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// GELU
static mllm_fp16_t mllm_table_gelu_f16[1 << 16];
static bool init_table_gelu_f16_flag = false;
inline void init_table_gelu_f16() {
    mllm_fp16_t ii;
    for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = MLLM_COMPUTE_FP16_TO_FP32(ii);
        mllm_table_gelu_f16[i] = MLLM_FP32_TO_FP16(mllm_gelu_f32(f));
    }
}
inline static void mllm_vec_gelu_f32(const int n, float *y, const float *x) {
    uint16_t t;
    // #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < n; ++i) {
        mllm_fp16_t fp16 = MLLM_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = MLLM_FP16_TO_FP32(mllm_table_gelu_f16[t]);
    }
}

// QuickGELU
static mllm_fp16_t mllm_table_gelu_quick_f16[1 << 16];
static bool init_table_gelu_quick_f16_flag = false;
inline void init_table_gelu_quick_f16() {
    mllm_fp16_t ii;
    for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = MLLM_COMPUTE_FP16_TO_FP32(ii);
        mllm_table_gelu_quick_f16[i] = MLLM_FP32_TO_FP16(mllm_gelu_quick_f32(f));
    }
}
inline static void mllm_vec_gelu_quick_f32(const int n, float *y, const float *x) {
    uint16_t t;
    // #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < n; ++i) {
        mllm_fp16_t fp16 = MLLM_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = MLLM_FP16_TO_FP32(mllm_table_gelu_quick_f16[t]);
    }
}
// SiLU
static mllm_fp16_t mllm_table_silu_f16[1 << 16];
static bool init_table_silu_f16_flag = false;
inline void init_table_silu_f16() {
    mllm_fp16_t ii;
    for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = MLLM_COMPUTE_FP16_TO_FP32(ii);
        mllm_table_silu_f16[i] = MLLM_FP32_TO_FP16(mllm_silu_f32(f));
    }
}
// inline static void mllm_vec_silu_f32(const int n, float * y, const float * x) {
//     uint16_t t;
// //#pragma omp parallel for num_threads(thread_count)
//     for (int i = 0; i < n; ++i) {
//         mllm_fp16_t fp16 = MLLM_FP32_TO_FP16(x[i]);
//         memcpy(&t, &fp16, sizeof(uint16_t));
//         y[i] = MLLM_FP16_TO_FP32(mllm_table_silu_f16[t]);
//     }
// }

#if __AVX__ || __AVX2__ || defined(__AVX512F__)
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t KShuffle[256] = {
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
    return _mm256_loadu_si256((const __m256i *)KShuffle + i);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    return _mm_loadu_si128((const __m128i *)k_shuffle + i);
}
#endif

static inline int nearest_int(float fval) {
    assert(fval <= 4194303.F);
    float val = fval + 12582912.F;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float *__restrict x, int8_t *__restrict L, int rmse_type) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < 1e-30f) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
        }
        return 1 / iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    int weight_type = rmse_type % 2;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax - 1, l));
        L[i] = l + nmax;
        float w = weight_type == 1 ? x[i] * x[i] : 1;
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = sumlx / suml2;
    if (return_early) return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax - 1, l));
            float w = weight_type == 1 ? x[i] * x[i] : 1;
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

// FP32_FP16

inline mllm_fp16_t mllm_fp32_to_fp16(float x) {
    return MLLM_FP32_TO_FP16(x);
}

inline float mllm_fp16_to_fp32(mllm_fp16_t x) {
    return (float)MLLM_FP16_TO_FP32(x);
}

inline void mllm_fp16_to_fp32_row(const mllm_fp16_t *x, float *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = MLLM_FP16_TO_FP32(x[i]);
    }
}

inline void mllm_fp32_to_fp16_row(const float *x, mllm_fp16_t *y, int n) {
    int i = 0;
#if defined(__F16C__)
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
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
