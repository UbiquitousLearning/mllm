//
// Created by lx on 23-11-15.
//

#ifndef MLLM_VECSCALAR_HPP
#define MLLM_VECSCALAR_HPP
#include <quantize/Quantize.hpp>
#include "Neon.hpp"
namespace mllm {
static inline void vec_scalar_fp32(const float *__restrict src0, const float scalar, float *dst, int n) {
#ifdef __ARM_NEON
    const int np = (n & ~(F32_STEP - 1));

    F32_VEC vx = vdupq_n_f32(scalar);

    F32_VEC ay[F32_ARR];
    for (int i = 0; i < np; i += F32_STEP) {
        for (int j = 0; j < F32_ARR; j++) {
            ay[j] = vld1q_f32(src0 + i + j * F32_REG);
            ay[j] = vmulq_f32(ay[j], vx);
            vst1q_f32(dst + i + j * F32_REG, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        dst[i] *= scalar;
    }
#else
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i] * v;
    }
#endif
}
static inline void vec_scalar_fp32_(float *vec, const float scalar, int n) {
#ifdef __ARM_NEON
    const int np = (n & ~(F32_STEP - 1));

    F32_VEC vx = vdupq_n_f32(scalar);

    F32_VEC ay[F32_ARR];
    for (int i = 0; i < np; i += F32_STEP) {
        for (int j = 0; j < F32_ARR; j++) {
            ay[j] = vld1q_f32(vec + i + j * F32_REG);
            ay[j] = vmulq_f32(ay[j], vx);
            vst1q_f32(vec + i + j * F32_REG, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        vec[i] *= scalar;
    }
#else
    for (int i = 0; i < n; ++i) {
        vec[i] *= v;
    }
#endif
};

} // namespace mllm

#endif // MLLM_VECSCALAR_HPP
