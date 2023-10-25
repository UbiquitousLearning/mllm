//
// Created by lx on 23-10-24.
//

#ifndef MLLM_NEON_HPP
#define MLLM_NEON_HPP

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
#endif

#endif // MLLM_NEON_HPP
