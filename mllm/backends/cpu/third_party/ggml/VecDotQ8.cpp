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

#include "VecDotQ8.hpp"
#include "ComputeUtils.hpp"

void vec_dot_q8_0_q8_0(int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy, size_t bs, size_t bx, size_t by) {
    const int qk = QK8_0;
    const int nb = n / qk; // number of blocks

    assert(n % qk == 0);

    const auto *__restrict x = static_cast<const block_q8_0 *>(vx);
    const auto *__restrict y = static_cast<const block_q8_0 *>(vy);

#if defined(__ARM_FEATURE_MATMUL_INT8)
    // if (nrc == 2)
    {
        const block_q8_0 *__restrict vx0 = (const block_q8_0 *)vx;
        const block_q8_0 *__restrict vx1 = (const block_q8_0 *)((const uint8_t *)vx + bx);
        const block_q8_0 *__restrict vy0 = (const block_q8_0 *)vy;
        const block_q8_0 *__restrict vy1 = (const block_q8_0 *)((const uint8_t *)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q8_0 *__restrict b_x0 = &vx0[i];
            const block_q8_0 *__restrict b_y0 = &vy0[i];

            const block_q8_0 *__restrict b_x1 = &vx1[i];
            const block_q8_0 *__restrict b_y1 = &vy1[i];

            const int8x16_t x0_l = vld1q_s8(b_x0->qs);
            const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
            const int8x16_t x1_l = vld1q_s8(b_x1->qs);
            const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                MLLM_FP16_TO_FP32(b_x0->d) * MLLM_FP16_TO_FP32(b_y0->d),
                MLLM_FP16_TO_FP32(b_x0->d) * MLLM_FP16_TO_FP32(b_y1->d),
                MLLM_FP16_TO_FP32(b_x1->d) * MLLM_FP16_TO_FP32(b_y0->d),
                MLLM_FP16_TO_FP32(b_x1->d) * MLLM_FP16_TO_FP32(b_y1->d)};
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0, (vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)), l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s, vget_low_f32(sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    assert(nb % 2 == 0); // TODO: handle odd nb

    for (int i = 0; i < nb; i += 2) {
        const block_q8_0 *x0 = &x[i + 0];
        const block_q8_0 *x1 = &x[i + 1];
        const block_q8_0 *y0 = &y[i + 0];
        const block_q8_0 *y1 = &y[i + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(mllm_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0), mllm_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), MLLM_FP16_TO_FP32(x0->d) * MLLM_FP16_TO_FP32(y0->d));

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(mllm_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0), mllm_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), MLLM_FP16_TO_FP32(x1->d) * MLLM_FP16_TO_FP32(y1->d));
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__) || defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(MLLM_FP16_TO_FP32(x[i].d) * MLLM_FP16_TO_FP32(y[i].d));
        __m256i bx = _mm256_loadu_si256((const __m256i *)x[i].qs);
        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        // Multiply q with scale and accumulate
#if defined(__AVX2__)
        acc = _mm256_fmadd_ps(d, q, acc);
#else
        acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
#endif
    }

    *s = hsum_float_8(acc);
#endif
}

void vec_dot_i8_i8(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy, float scale1, float scale2) {
    const int qk = QK8_0;
    const int nb = n / qk;

    const float scale = scale1 * scale2;

    assert(n % qk == 0);

    const block_q8_per_tensor *__restrict x = (block_q8_per_tensor *)vx;
    const block_q8_per_tensor *__restrict y = (block_q8_per_tensor *)vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    assert(nb % 2 == 0); // TODO: handle odd nb

    for (int i = 0; i < nb; i += 2) {
        const block_q8_per_tensor *__restrict x0 = &x[i + 0];
        const block_q8_per_tensor *__restrict x1 = &x[i + 1];
        const block_q8_per_tensor *__restrict y0 = &y[i + 0];
        const block_q8_per_tensor *__restrict y1 = &y[i + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(mllm_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0), mllm_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), scale);

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(mllm_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0), mllm_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), scale);
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__) || defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(scale);
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[i].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
#if defined(__AVX2__)
        acc = _mm256_fmadd_ps(d, q, acc);
#else
        acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
#endif
    }

    *s = hsum_float_8(acc);
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[i].qs[j] * y[i].qs[j];
        }

        sumf += sumi * scale;
    }

    *s = sumf;
#endif
}