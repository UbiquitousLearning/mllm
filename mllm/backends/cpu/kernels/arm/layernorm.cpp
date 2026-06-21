// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/layernorm.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

void layernorm_N_fp32(mllm_fp32_t* __restrict__ Z, const mllm_fp32_t* __restrict__ X, const mllm_fp32_t* __restrict__ gamma,
                      const mllm_fp32_t* __restrict__ beta, size_t N, mllm_fp32_t eps, int32_t thread_count) {
  if (N == 0) return;

  float sum = 0.0f;
  size_t i = 0;
  float32x4_t vsum = vdupq_n_f32(0.0f);

  for (; i + 3 < N; i += 4) {
    float32x4_t vx = vld1q_f32(X + i);
    vsum = vaddq_f32(vsum, vx);
  }

  sum = vaddvq_f32(vsum);

  for (; i < N; i++) { sum += X[i]; }

  const float mean = sum / N;
  const float32x4_t vmean = vdupq_n_f32(mean);

  float sq_sum = 0.0f;
  i = 0;
  float32x4_t vsq_sum = vdupq_n_f32(0.0f);

  for (; i + 3 < N; i += 4) {
    float32x4_t vx = vld1q_f32(X + i);
    float32x4_t vdiff = vsubq_f32(vx, vmean);
    vsq_sum = vmlaq_f32(vsq_sum, vdiff, vdiff);  // vsq_sum += vdiff * vdiff
  }

  sq_sum = vaddvq_f32(vsq_sum);

  for (; i < N; i++) {
    float diff = X[i] - mean;
    sq_sum += diff * diff;
  }

  const float variance = sq_sum / N;
  const float std_val = 1.0f / sqrtf(variance + eps);
  const float32x4_t vscale = vdupq_n_f32(std_val);

  i = 0;
  if (gamma && beta) {
    for (; i + 3 < N; i += 4) {
      float32x4_t vx = vld1q_f32(X + i);
      float32x4_t vdiff = vsubq_f32(vx, vmean);
      float32x4_t vnorm = vmulq_f32(vdiff, vscale);

      float32x4_t vgamma = vld1q_f32(gamma + i);
      float32x4_t vbeta = vld1q_f32(beta + i);

      float32x4_t vz = vmlaq_f32(vbeta, vnorm, vgamma);
      vst1q_f32(Z + i, vz);
    }
  } else if (gamma) {
    for (; i + 3 < N; i += 4) {
      float32x4_t vx = vld1q_f32(X + i);
      float32x4_t vdiff = vsubq_f32(vx, vmean);
      float32x4_t vnorm = vmulq_f32(vdiff, vscale);

      float32x4_t vgamma = vld1q_f32(gamma + i);
      float32x4_t vz = vmulq_f32(vnorm, vgamma);
      vst1q_f32(Z + i, vz);
    }
  } else if (beta) {
    for (; i + 3 < N; i += 4) {
      float32x4_t vx = vld1q_f32(X + i);
      float32x4_t vdiff = vsubq_f32(vx, vmean);
      float32x4_t vnorm = vmulq_f32(vdiff, vscale);

      float32x4_t vbeta = vld1q_f32(beta + i);
      float32x4_t vz = vaddq_f32(vnorm, vbeta);
      vst1q_f32(Z + i, vz);
    }
  } else {
    for (; i + 3 < N; i += 4) {
      float32x4_t vx = vld1q_f32(X + i);
      float32x4_t vdiff = vsubq_f32(vx, vmean);
      float32x4_t vz = vmulq_f32(vdiff, vscale);
      vst1q_f32(Z + i, vz);
    }
  }

  for (; i < N; i++) {
    float norm_val = (X[i] - mean) * std_val;
    if (gamma) norm_val *= gamma[i];
    if (beta) norm_val += beta[i];
    Z[i] = norm_val;
  }
}

void layernorm_N_fp16(mllm_fp16_t* __restrict__ Z, const mllm_fp16_t* __restrict__ X, const mllm_fp16_t* __restrict__ gamma,
                      const mllm_fp16_t* __restrict__ beta, size_t N, mllm_fp32_t eps, int32_t thread_count) {
  if (N == 0) return;

  // Use float to keep accuracy.
  float sum = 0.0f;
  size_t i = 0;

  float32x4_t vsum_low = vdupq_n_f32(0.0f);
  float32x4_t vsum_high = vdupq_n_f32(0.0f);

  for (; i + 7 < N; i += 8) {
    float16x8_t vx = vld1q_f16(X + i);
    vsum_low = vaddq_f32(vsum_low, vcvt_f32_f16(vget_low_f16(vx)));
    vsum_high = vaddq_f32(vsum_high, vcvt_f32_f16(vget_high_f16(vx)));
  }
  sum += vaddvq_f32(vsum_low) + vaddvq_f32(vsum_high);

  for (; i < N; i++) { sum += (float)X[i]; }

  const float mean = sum / N;
  const __fp16 mean_fp16 = (__fp16)mean;

  float sq_sum = 0.0f;
  i = 0;
  float32x4_t vsq_sum_low = vdupq_n_f32(0.0f);
  float32x4_t vsq_sum_high = vdupq_n_f32(0.0f);
  const float32x4_t vmean = vdupq_n_f32(mean);

  for (; i + 7 < N; i += 8) {
    float16x8_t vx = vld1q_f16(X + i);
    float32x4_t vx_low = vcvt_f32_f16(vget_low_f16(vx));
    float32x4_t vx_high = vcvt_f32_f16(vget_high_f16(vx));

    float32x4_t vdiff_low = vsubq_f32(vx_low, vmean);
    float32x4_t vdiff_high = vsubq_f32(vx_high, vmean);

    vsq_sum_low = vfmaq_f32(vsq_sum_low, vdiff_low, vdiff_low);
    vsq_sum_high = vfmaq_f32(vsq_sum_high, vdiff_high, vdiff_high);
  }
  sq_sum += vaddvq_f32(vsq_sum_low) + vaddvq_f32(vsq_sum_high);

  for (; i < N; i++) {
    float diff = (float)X[i] - mean;
    sq_sum += diff * diff;
  }

  const float variance = sq_sum / N;
  const float std_val = 1.0f / sqrtf(variance + eps);
  const __fp16 std_val_fp16 = (__fp16)std_val;

  i = 0;
  const float16x8_t vmean_fp16 = vdupq_n_f16(mean_fp16);
  const float16x8_t vscale = vdupq_n_f16(std_val_fp16);

  if (gamma && beta) {
    for (; i + 7 < N; i += 8) {
      float16x8_t vx = vld1q_f16(X + i);
      float16x8_t vdiff = vsubq_f16(vx, vmean_fp16);
      float16x8_t vnorm = vmulq_f16(vdiff, vscale);

      float16x8_t vgamma = vld1q_f16(gamma + i);
      float16x8_t vbeta = vld1q_f16(beta + i);

      // Z = norm * gamma + beta
      float16x8_t vz = vfmaq_f16(vbeta, vnorm, vgamma);
      vst1q_f16(Z + i, vz);
    }
  } else if (gamma) {
    for (; i + 7 < N; i += 8) {
      float16x8_t vx = vld1q_f16(X + i);
      float16x8_t vdiff = vsubq_f16(vx, vmean_fp16);
      float16x8_t vnorm = vmulq_f16(vdiff, vscale);

      float16x8_t vgamma = vld1q_f16(gamma + i);
      vst1q_f16(Z + i, vmulq_f16(vnorm, vgamma));
    }
  } else if (beta) {
    for (; i + 7 < N; i += 8) {
      float16x8_t vx = vld1q_f16(X + i);
      float16x8_t vdiff = vsubq_f16(vx, vmean_fp16);
      float16x8_t vnorm = vmulq_f16(vdiff, vscale);

      float16x8_t vbeta = vld1q_f16(beta + i);
      vst1q_f16(Z + i, vaddq_f16(vnorm, vbeta));
    }
  } else {
    for (; i + 7 < N; i += 8) {
      float16x8_t vx = vld1q_f16(X + i);
      float16x8_t vdiff = vsubq_f16(vx, vmean_fp16);
      vst1q_f16(Z + i, vmulq_f16(vdiff, vscale));
    }
  }

  for (; i < N; i++) {
    float16_t norm_val = (X[i] - mean_fp16) * std_val_fp16;
    if (gamma) norm_val *= gamma[i];
    if (beta) norm_val += beta[i];
    Z[i] = norm_val;
  }
}

}  // namespace mllm::cpu::arm

#endif
