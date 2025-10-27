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

#include "VecDotFP32.hpp"

#ifdef __AVX2__
static void vec_dot_fp32_avx2(const int n, float *__restrict s, const float *__restrict x, const float *__restrict y) {
    float sumf = 0.0F;
    const int np = (n & ~(MLLM_F32_STEP - 1));

    MLLM_F32_VEC sum[MLLM_F32_ARR] = {MLLM_F32_VEC_ZERO};

    MLLM_F32_VEC ax[MLLM_F32_ARR];
    MLLM_F32_VEC ay[MLLM_F32_ARR];

    for (int i = 0; i < np; i += MLLM_F32_STEP) {
        for (int j = 0; j < MLLM_F32_ARR; j++) {
            ax[j] = MLLM_F32_VEC_LOAD(x + i + j * MLLM_F32_EPR);
            ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);

            sum[j] = MLLM_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    MLLM_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }

    *s = sumf;
}
#endif

#ifdef __ARM_NEON
static void vec_dot_fp32_arm(const int n, float *__restrict s, const float *__restrict x, const float *__restrict y) {
    float sumf = 0.0F;
    const int np = (n & ~(16 - 1));

    F32_VEC sum[4] = {vdupq_n_f32(0.0F)};

    F32_VEC ax[F32_ARR];
    F32_VEC ay[F32_ARR];

    for (int i = 0; i < np; i += F32_STEP) {
        for (int j = 0; j < F32_ARR; j++) {
            ax[j] = vld1q_f32(x + i + j * F32_REG);
            ay[j] = vld1q_f32(y + i + j * F32_REG);
            sum[j] = vfmaq_f32(sum[j], ax[j], ay[j]);
            // sum[j] = vmlaq_lane_f32(sum[j], ax[j], ay[0],
        }
    }

    // reduce sum0..sum3 to sum0
    F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }

    *s = sumf;
}
#endif

void vec_dot_fp32(const int n, float *__restrict s, const float *__restrict vx, const float *__restrict vy) {
#ifdef __AVX2__
    vec_dot_fp32_avx2(n, s, vx, vy);
#elif defined(__ARM_NEON)
    vec_dot_fp32_arm(n, s, vx, vy);
#endif
}

#ifdef __AVX2__
static void vec_value_dot_fp32_avx2(const int n, float *__restrict s, const float *__restrict x, const float *__restrict y, bool addition) {
    float sumf = 0.0F;
    const int np = (n & ~(MLLM_F32_STEP - 1));

    MLLM_F32_VEC sum[MLLM_F32_ARR] = {MLLM_F32_VEC_ZERO};

    MLLM_F32_VEC ax[MLLM_F32_ARR];
    MLLM_F32_VEC ay[MLLM_F32_ARR];

    for (int i = 0; i < np; i += MLLM_F32_STEP) {
        for (int j = 0; j < MLLM_F32_ARR; j++) {
            ax[j] = MLLM_F32_VEC_LOAD(x + i + j * MLLM_F32_EPR);
            ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);

            sum[j] = MLLM_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    MLLM_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }

    *s = sumf;
}
#endif

#ifdef __ARM_NEON
// s:vector k
// x:value
// y:vector k
static void vec_value_dot_fp32_arm(const int n, float *__restrict s, const float x, const float *__restrict y, bool addition) {
    int i;
    float32x4_t vec_x;
    float32x4_t vec_y;
    float32x4_t vec_s;

    vec_x = vdupq_n_f32(x);

    int n_aligned = n & -4;

    if (addition) {
        for (i = 0; i < n_aligned; i += 4) {
            vec_y = vld1q_f32(y + i);
            vec_s = vmulq_f32(vec_x, vec_y);
            vec_s = vaddq_f32(vec_s, vld1q_f32(s + i));
            vst1q_f32(s + i, vec_s);
        }
    } else {
        for (i = 0; i < n_aligned; i += 4) {
            vec_y = vld1q_f32(y + i);
            vec_s = vmulq_f32(vec_x, vec_y);
            vst1q_f32(s + i, vec_s);
        }
    }
    for (; i < n; ++i) {
        if (addition)
            s[i] += x * y[i];
        else {
            s[i] = x * y[i];
        }
    }
}
#endif

#ifdef __AVX2__
void vec_value_dot_fp32(const int n, float *__restrict s, const float *x, const float *__restrict vy, bool addition) {
    vec_value_dot_fp32_avx2(n, s, x, vy, addition);
}
#elif defined(__ARM_NEON)
void vec_value_dot_fp32(const int n, float *__restrict s, const float x, const float *__restrict vy, bool addition) {
    vec_value_dot_fp32_arm(n, s, x, vy, addition);
}
#endif
