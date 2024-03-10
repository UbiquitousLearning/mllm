//
// Created by shrelic on 24-3-5.
//
#include <cstdio>
#include <cstring>
#include "type.hpp"
#include "Types.hpp"
#include "quantize/Quantize.hpp"
#include "compute/VecDot.hpp"

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

static void vec_dot_fp32_local(const int n, float *__restrict s, const float *__restrict vx, const float *__restrict vy) {
#ifdef __AVX2__
    vec_dot_fp32_avx2(n, s, vx, vy);
#elif defined(__ARM_NEON)
    vec_dot_fp32_arm(n, s, vx, vy);
#endif
}

void fp32_add_row_to(int n, const float * MLLM_RESTRICT src, float * MLLM_RESTRICT dst, float alpha){
    int i = 0;
#ifdef __AVX2__
    __m256 alpha_vec = _mm256_set1_ps(alpha); // load alpha into 8 float register

    // 主循环处理8的倍数个元素
    for (; i <= n - 8; i += 8) {
        __m256 src_vec = _mm256_loadu_ps(src + i); // load 8 float from src
        __m256 dst_vec = _mm256_loadu_ps(dst + i); // load 8 float from dst
        __m256 res_vec = _mm256_fmadd_ps(src_vec, alpha_vec, dst_vec); // alpha * src + dst
        _mm256_storeu_ps(dst + i, res_vec); // store back to dst
    }
#elif defined(__ARM_NEON)
    // TODO: generated by GPT-4, not tested yet
    float32x4_t alpha_vec = vdupq_n_f32(alpha); // load alpha into all elements of a 128-bit register

    // Main loop for multiples of 4
    for (; i <= n - 4; i += 4) {
        float32x4_t src_vec = vld1q_f32(src + i);
        float32x4_t dst_vec = vld1q_f32(dst + i);
        float32x4_t res_vec = vmlaq_f32(dst_vec, src_vec, alpha_vec); // calculate alpha * src + dst
        vst1q_f32(dst + i, res_vec); // store result back to dst
    }
#endif

    // 处理剩余的元素
    for (; i < n; ++i) {
        dst[i] = dst[i] + alpha * src[i];
    }
}

void fp_16_add_row_to(int n, const mllm_fp16_t * MLLM_RESTRICT src, float * MLLM_RESTRICT dst, float alpha){
    int i = 0;
#ifdef __AVX2__
    __m256 alpha_vec = _mm256_set1_ps(alpha); // load alpha into 8 float register

    // 主循环处理8的倍数个元素
    for (; i <= n - 8; i += 8) {
        __m128i src_fp16 = _mm_loadu_si128((__m128i const*)(src + i)); // load 8 fp16 from src
        __m256 src_vec = _mm256_cvtph_ps(src_fp16); // convert to 8 fp32
        __m256 dst_vec = _mm256_loadu_ps(dst + i);  // load 8 float from dst
        __m256 res_vec = _mm256_fmadd_ps(src_vec, alpha_vec, dst_vec); // alpha * src + dst
        _mm256_storeu_ps(dst + i, res_vec); // store back to dst
    }
#elif defined(__ARM_NEON)
    ASSERT(false); // not support now
#endif

    // 处理剩余的元素
    for (; i < n; ++i) {
        dst[i] = dst[i] + alpha * MLLM_FP16_TO_FP32(src[i]);
    }
}

type_traits_t type_traits[] = {
    /*[MLLM_TYPE_F32] = */{
        .size = sizeof(float),
        .to_float = nullptr,
        .from_float = nullptr,
        .vec_dot = (mllm_vec_dot_func)vec_dot_fp32_local,
        .vec_dot_type = MLLM_TYPE_F32,
        .add_row_to = (mllm_vec_add_row_func)fp32_add_row_to,
    },
    /*[MLLM_TYPE_F16] = */{
        .size = sizeof(mllm_fp16_t),
        .to_float = (mllm_to_float_func)mllm_fp16_to_fp32_row,
        .from_float = (mllm_from_float_func)mllm_fp32_to_fp16_row,
        .vec_dot = (mllm_vec_dot_func)vec_dot_fp16,
        .vec_dot_type = MLLM_TYPE_F16,
        .add_row_to = (mllm_vec_add_row_func)fp_16_add_row_to,
    }
    // TODO: add support to more type
};