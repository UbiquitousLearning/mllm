// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/softmax.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>
#include <arm_fp16.h>

#include "mllm/backends/cpu/kernels/arm/math.hpp"

namespace mllm::cpu::arm {

// Safe sofmax for fp32. Not optimized for stride!=1 situation. When stride is set to 1, this
// function will utilize vexp1_fast_fp32 method to accelerate exp computation. This function not
// required (len % K == 0), any length is acceptable.
void softmax_v1_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int stride, int thread_count) {
  // memory is not continue.
  if (stride != 1 || len <= 16) {
    float max_value = std::numeric_limits<float>::lowest();
    // Pass 1: find the max value in X
    for (int i = 0; i < len; ++i) { max_value = std::max(max_value, X[i * stride]); }

    // Pass 2: minus max_value and calculate exp
    float sum = 0.f;
    for (int i = 0; i < len; ++i) {
      auto tmp = std::expf(X[i * stride] - max_value);
      Y[i * stride] = tmp;
      sum += tmp;
    }

    sum = 1.f / sum;

    // Pass 3: divide sum
    for (int i = 0; i < len; ++i) { Y[i * stride] *= sum; }

    return;
  }

  // use vectorized version when stride is 1 and len is large.
  // Pass 1: find the max value in X
  float32x4_t max_vec_0 = vld1q_f32(X);
  float32x4_t max_vec_1 = vld1q_f32(X + 4);
  float32x4_t max_vec_2 = vld1q_f32(X + 8);
  float32x4_t max_vec_3 = vld1q_f32(X + 12);
  int i;
  for (i = 16; i <= len - 16; i += 16) {
    float32x4_t tmp_0 = vld1q_f32(X + i);
    max_vec_0 = vmaxq_f32(max_vec_0, tmp_0);

    float32x4_t tmp_1 = vld1q_f32(X + i + 4);
    max_vec_1 = vmaxq_f32(max_vec_1, tmp_1);

    float32x4_t tmp_2 = vld1q_f32(X + i + 8);
    max_vec_2 = vmaxq_f32(max_vec_2, tmp_2);

    float32x4_t tmp_3 = vld1q_f32(X + i + 12);
    max_vec_3 = vmaxq_f32(max_vec_3, tmp_3);
  }
  max_vec_0 = vmaxq_f32(max_vec_0, max_vec_1);
  max_vec_2 = vmaxq_f32(max_vec_2, max_vec_3);
  max_vec_0 = vmaxq_f32(max_vec_0, max_vec_2);

  for (; i <= len - 4; i += 4) {
    float32x4_t tmp_0 = vld1q_f32(X + i);
    max_vec_0 = vmaxq_f32(max_vec_0, tmp_0);
  }
  float max_value = vmaxvq_f32(max_vec_0);
  for (; i < len; ++i) { max_value = std::max(max_value, X[i]); }

  // Pass 2: minus max_value and calculate exp and sumup
  float32x4_t sum_vec_0 = vdupq_n_f32(0.f);
  float32x4_t sum_vec_1 = vdupq_n_f32(0.f);
  float32x4_t sum_vec_2 = vdupq_n_f32(0.f);
  float32x4_t sum_vec_3 = vdupq_n_f32(0.f);

  max_vec_0 = vdupq_n_f32(max_value);
  max_vec_1 = vdupq_n_f32(max_value);
  max_vec_2 = vdupq_n_f32(max_value);
  max_vec_3 = vdupq_n_f32(max_value);

  for (i = 0; i <= len - 16; i += 16) {
    float32x4_t tmp_0 = vld1q_f32(X + i);
    float32x4_t exp_tmp_0 = vexpq_fast_f32(vsubq_f32(tmp_0, max_vec_0));
    sum_vec_0 = vaddq_f32(sum_vec_0, exp_tmp_0);
    vst1q_f32(Y + i, exp_tmp_0);

    float32x4_t tmp_1 = vld1q_f32(X + i + 4);
    float32x4_t exp_tmp_1 = vexpq_fast_f32(vsubq_f32(tmp_1, max_vec_1));
    sum_vec_1 = vaddq_f32(sum_vec_1, exp_tmp_1);
    vst1q_f32(Y + i + 4, exp_tmp_1);

    float32x4_t tmp_2 = vld1q_f32(X + i + 8);
    float32x4_t exp_tmp_2 = vexpq_fast_f32(vsubq_f32(tmp_2, max_vec_2));
    sum_vec_2 = vaddq_f32(sum_vec_2, exp_tmp_2);
    vst1q_f32(Y + i + 8, exp_tmp_2);

    float32x4_t tmp_3 = vld1q_f32(X + i + 12);
    float32x4_t exp_tmp_3 = vexpq_fast_f32(vsubq_f32(tmp_3, max_vec_3));
    sum_vec_3 = vaddq_f32(sum_vec_3, exp_tmp_3);
    vst1q_f32(Y + i + 12, exp_tmp_3);
  }
  sum_vec_0 = vaddq_f32(sum_vec_0, sum_vec_1);
  sum_vec_2 = vaddq_f32(sum_vec_2, sum_vec_3);
  sum_vec_0 = vaddq_f32(sum_vec_0, sum_vec_2);
  for (; i <= len - 4; i += 4) {
    float32x4_t tmp_0 = vld1q_f32(X + i);
    float32x4_t exp_tmp_0 = vexpq_fast_f32(vsubq_f32(tmp_0, max_vec_0));
    sum_vec_0 = vaddq_f32(sum_vec_0, exp_tmp_0);
    vst1q_f32(Y + i, exp_tmp_0);
  }
  float sum_value = vaddvq_f32(sum_vec_0);
  for (; i < len; ++i) {
    float tmp = std::expf(X[i] - max_value);
    Y[i] = tmp;
    sum_value += tmp;
  }
  sum_value = 1.f / sum_value;

  // Pass 3: divide sum
  sum_vec_0 = vdupq_n_f32(sum_value);
  sum_vec_1 = vdupq_n_f32(sum_value);
  sum_vec_2 = vdupq_n_f32(sum_value);
  sum_vec_3 = vdupq_n_f32(sum_value);

  for (i = 0; i <= len - 16; i += 16) {
    float32x4_t tmp_0 = vld1q_f32(Y + i);
    float32x4_t ans_0 = vmulq_f32(tmp_0, sum_vec_0);
    vst1q_f32(Y + i, ans_0);

    float32x4_t tmp_1 = vld1q_f32(Y + i + 4);
    float32x4_t ans_1 = vmulq_f32(tmp_1, sum_vec_1);
    vst1q_f32(Y + i + 4, ans_1);

    float32x4_t tmp_2 = vld1q_f32(Y + i + 8);
    float32x4_t ans_2 = vmulq_f32(tmp_2, sum_vec_2);
    vst1q_f32(Y + i + 8, ans_2);

    float32x4_t tmp_3 = vld1q_f32(Y + i + 12);
    float32x4_t ans_3 = vmulq_f32(tmp_3, sum_vec_3);
    vst1q_f32(Y + i + 12, ans_3);
  }
  for (; i <= len - 4; i += 4) {
    float32x4_t tmp_0 = vld1q_f32(Y + i);
    float32x4_t ans_0 = vmulq_f32(tmp_0, sum_vec_0);
    vst1q_f32(Y + i, ans_0);
  }
  for (; i < len; ++i) { Y[i] *= sum_value; }
}

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else

void softmax_v1_fp16(const mllm_fp16_t* __restrict X, mllm_fp16_t* __restrict Y, int len, int stride, int thread_count) {
  // memory is not continue.
  if (stride != 1 || len <= 16) {
    float16_t max_value = std::numeric_limits<float16_t>::lowest();
    // Pass 1: find the max value in X
    for (int i = 0; i < len; ++i) { max_value = vmaxh_f16(max_value, X[i * stride]); }

    // Pass 2: minus max_value and calculate exp
    float sum = 0.f;
    for (int i = 0; i < len; ++i) {
      auto tmp = std::expf(X[i * stride] - max_value);
      Y[i * stride] = static_cast<float16_t>(tmp);
      sum += tmp;
    }

    sum = 1.f / sum;

    // Pass 3: divide sum
    for (int i = 0; i < len; ++i) { Y[i * stride] = vmulh_f16(sum, Y[i * stride]); }

    return;
  }

  // use vectorized version when stride is 1 and len is large.
  // Pass 1: find the max value in X
  float16x8_t max_vec_0 = vld1q_f16(X);
  float16x8_t max_vec_1 = vld1q_f16(X + 8);
  int i;
  for (i = 16; i <= len - 16; i += 16) {
    float16x8_t tmp_0 = vld1q_f16(X + i);
    max_vec_0 = vmaxq_f16(max_vec_0, tmp_0);

    float16x8_t tmp_1 = vld1q_f16(X + i + 8);
    max_vec_1 = vmaxq_f16(max_vec_1, tmp_1);
  }
  max_vec_0 = vmaxq_f16(max_vec_0, max_vec_1);
  for (; i <= len - 8; i += 8) {
    float16x8_t tmp_0 = vld1q_f16(X + i);
    max_vec_0 = vmaxq_f16(max_vec_0, tmp_0);
  }
  float16_t max_value = vmaxvq_f16(max_vec_0);
  for (; i < len; ++i) { max_value = vmaxh_f16(max_value, X[i]); }

  // Pass 2: minus max_value and calculate exp and sumup
  float32x4_t sum_vec_0 = vdupq_n_f32(0.f);
  float32x4_t sum_vec_1 = vdupq_n_f32(0.f);
  float32x4_t sum_vec_2 = vdupq_n_f32(0.f);
  float32x4_t sum_vec_3 = vdupq_n_f32(0.f);

  max_vec_0 = vdupq_n_f16(max_value);
  max_vec_1 = vdupq_n_f16(max_value);
  for (i = 0; i <= len - 16; i += 16) {
    float16x8_t tmp_0 = vld1q_f16(X + i);
    float16x8_t exp_tmp_0 = vexpq_fast_f16(vsubq_f16(tmp_0, max_vec_0));
    sum_vec_0 = vaddq_f32(sum_vec_0, vcvt_f32_f16(vget_high_f16(exp_tmp_0)));
    sum_vec_1 = vaddq_f32(sum_vec_1, vcvt_f32_f16(vget_low_f16(exp_tmp_0)));
    vst1q_f16(Y + i, exp_tmp_0);

    float16x8_t tmp_1 = vld1q_f16(X + i + 8);
    float16x8_t exp_tmp_1 = vexpq_fast_f16(vsubq_f16(tmp_1, max_vec_1));
    sum_vec_2 = vaddq_f32(sum_vec_2, vcvt_f32_f16(vget_high_f16(exp_tmp_1)));
    sum_vec_3 = vaddq_f32(sum_vec_3, vcvt_f32_f16(vget_low_f16(exp_tmp_1)));
    vst1q_f16(Y + i + 8, exp_tmp_1);
  }
  for (; i <= len - 8; i += 8) {
    float16x8_t tmp_0 = vld1q_f16(X + i);
    float16x8_t exp_tmp_0 = vexpq_fast_f16(vsubq_f16(tmp_0, max_vec_0));
    sum_vec_0 = vaddq_f32(sum_vec_0, vcvt_f32_f16(vget_high_f16(exp_tmp_0)));
    sum_vec_1 = vaddq_f32(sum_vec_1, vcvt_f32_f16(vget_low_f16(exp_tmp_0)));
    vst1q_f16(Y + i, exp_tmp_0);
  }
  sum_vec_0 = vaddq_f32(sum_vec_0, sum_vec_1);
  sum_vec_2 = vaddq_f32(sum_vec_2, sum_vec_3);
  sum_vec_0 = vaddq_f32(sum_vec_0, sum_vec_2);
  float sum_value = vaddvq_f32(sum_vec_0);
  for (; i < len; ++i) {
    float tmp = std::expf(X[i] - max_value);
    Y[i] = static_cast<float16_t>(tmp);
    sum_value += tmp;
  }
  sum_value = 1.f / sum_value;

  // Pass 3: divide sum
  float16x8_t divide_sum_0 = vdupq_n_f16(static_cast<float16_t>(sum_value));
  float16x8_t divide_sum_1 = vdupq_n_f16(static_cast<float16_t>(sum_value));
  for (i = 0; i <= len - 16; i += 16) {
    float16x8_t tmp_0 = vld1q_f16(Y + i);
    float16x8_t ans_0 = vmulq_f16(tmp_0, divide_sum_0);
    vst1q_f16(Y + i, ans_0);

    float16x8_t tmp_1 = vld1q_f16(Y + i + 8);
    float16x8_t ans_1 = vmulq_f16(tmp_1, divide_sum_1);
    vst1q_f16(Y + i + 8, ans_1);
  }
  for (; i <= len - 8; i += 8) {
    float16x8_t tmp_0 = vld1q_f16(Y + i);
    float16x8_t ans_0 = vmulq_f16(tmp_0, divide_sum_0);
    vst1q_f16(Y + i, ans_0);
  }
  for (; i < len; ++i) { Y[i] = vmulh_f16(Y[i], sum_value); }
}

#endif  // fp16

}  // namespace mllm::cpu::arm

#endif