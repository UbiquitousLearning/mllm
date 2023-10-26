//
// Created by lx on 23-10-24.
//

#ifndef MLLM_NEON_HPP
#define MLLM_NEON_HPP
#include "../quantize/Quantize.hpp"
#include "assert.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
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
static void vec_dot_f32_arm(const int n, float *__restrict s, const float *__restrict x, const float *__restrict y) {
    float sumf = 0.0F;
    const int np = (n & ~(4 - 1));

    F32_VEC sum[4] = {vdupq_n_f32(0.0F)};

    F32_VEC ax[F32_ARR];
    F32_VEC ay[F32_ARR];

    for (int i = 0; i < np; i += F32_STEP) {
        for (int j = 0; j < F32_ARR; j++) {
            ax[j] = vld1q_f32(x + i + j * F32_REG);
            ay[j] = vld1q_f32(y + i + j * F32_REG);
            sum[j] = vmlaq_f32(sum[j], ax[j], ay[j]);
            // sum[j] = vmlaq_lane_f32(sum[j], ax[j], ay[0],
        }
    }

    // reduce sum0..sum3 to sum0
    F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }
}
// COPY FROMN
#define QK8_0 32
static void vec_dot_q4_0_q8_0_arm(const int n, float * __restrict s, const void * __restrict vx, const void * __restrict vy) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);

    const block_q4_0 *__restrict x = (block_q4_0 *)vx;
    const block_q8_0 *__restrict y = (block_q8_0 *)vy;
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    assert(nb % 2 == 0); // TODO: handle odd nb
    for (int i = 0; i < nb; i += 2) {
        const block_q4_0 * restrict x0 = &x[i + 0];
        const block_q4_0 * restrict x1 = &x[i + 1];
        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), MLLM_FP16_TO_FP32(x0->d)*MLLM_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), MLLM_FP16_TO_FP32(x1->d)*MLLM_FP16_TO_FP32(y1->d));
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), MLLM_FP16_TO_FP32(x0->d)*MLLM_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), MLLM_FP16_TO_FP32(x1->d)*MLLM_FP16_TO_FP32(y1->d));
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
}

#endif

#endif // MLLM_NEON_HPP
