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

#pragma once
#include "ComputeUtils.hpp"

// using namespace mllm;

inline static void vec_scale_f32(const int n, float *y, const float v) {
    const int np = (n & ~(MLLM_F32_STEP - 1));

    MLLM_F32_VEC vx = MLLM_F32_VEC_SET1(v);

    MLLM_F32_VEC ay[MLLM_F32_ARR];

    for (int i = 0; i < np; i += MLLM_F32_STEP) {
        for (int j = 0; j < MLLM_F32_ARR; j++) {
            ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);
            ay[j] = MLLM_F32_VEC_MUL(ay[j], vx);

            MLLM_F32_VEC_STORE(y + i + (j * MLLM_F32_EPR), ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
}

inline void vec_mul_fp32(const int n, float *__restrict s, const float *__restrict x, const float *__restrict y) {
    int i = 0;
    const int np = (n & ~(MLLM_F32_STEP - 1));
    MLLM_F32_VEC ax[MLLM_F32_ARR];
    MLLM_F32_VEC ay[MLLM_F32_ARR];
    MLLM_F32_VEC as[MLLM_F32_ARR];
    for (i = 0; i < np; i += MLLM_F32_STEP) {
        for (int j = 0; j < MLLM_F32_ARR; j++) {
            ax[j] = MLLM_F32_VEC_LOAD(x + i + j * MLLM_F32_EPR);
            ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);
            as[j] = MLLM_F32_VEC_MUL(ax[j], ay[j]);
            MLLM_F32_VEC_STORE(s + i + (j * MLLM_F32_EPR), as[j]);
        }
    }
    for (; i < n; ++i) {
        s[i] = x[i] * y[i];
    }
}

void vec_dot_fp32(const int n, float *__restrict s, const float *__restrict vx, const float *__restrict vy);
// for sparse linear
void vec_value_dot_fp32(const int n, float *__restrict s, const float x, const float *__restrict vy, bool addition);