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

#include "VecDotFP16.hpp"

void vec_dot_fp16(const int n, float *__restrict s, const mllm_fp16_t *__restrict vx, const mllm_fp16_t *__restrict vy) {
    float sumf = 0.0;

#if defined(__AVX2__) || defined(__ARM_NEON)
    const int np = (n & ~(MLLM_F16_STEP - 1));

    MLLM_F16_VEC sum[MLLM_F16_ARR] = {MLLM_F16_VEC_ZERO};

    MLLM_F16_VEC ax[MLLM_F16_ARR];
    MLLM_F16_VEC ay[MLLM_F16_ARR];

    for (int i = 0; i < np; i += MLLM_F16_STEP) {
        for (int j = 0; j < MLLM_F16_ARR; j++) {
            ax[j] = MLLM_F16_VEC_LOAD(vx + i + j * MLLM_F16_EPR, j);
            ay[j] = MLLM_F16_VEC_LOAD(vy + i + j * MLLM_F16_EPR, j);

            sum[j] = MLLM_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    MLLM_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (float)(MLLM_FP16_TO_FP32(vx[i]) * MLLM_FP16_TO_FP32(vy[i]));
    }
#else
    for (int i = 0; i < n; ++i) {
        sumf += (float)(MLLM_FP16_TO_FP32(vx[i]) * MLLM_FP16_TO_FP32(vy[i]));
    }
#endif

    *s = sumf;
}
