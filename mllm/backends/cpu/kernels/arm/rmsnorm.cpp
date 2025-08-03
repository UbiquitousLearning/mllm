// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/rmsnorm.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include "mllm/backends/cpu/kernels/arm/math.hpp"

namespace mllm::cpu::arm {

// Should support [B, S, H * D] and [B, S, H, D]
void rmsnorm_fp32(const mllm_fp32_t* __restrict X, const mllm_fp32_t* __restrict W, mllm_fp32_t* __restrict Y, int D,
                  float epsilon, bool add_unit_offset, int thread_count) {
  auto x_ptr = X;
  auto y_ptr = Y;
  auto w_ptr = W;

  // pass 1
  const float rms = 1.f / std::sqrtf(vsquare_mean_fp32(x_ptr, D) + epsilon);

  // pass 2
  if (add_unit_offset) {
    float32x4_t ones = vdupq_n_f32(1.f);
    int d;
    for (d = 0; d <= D - 4; ++d) {
      float32x4_t tmp_x = vld1q_f32(x_ptr + d);
      float32x4_t multiplier = vld1q_f32(w_ptr + d);
      multiplier = vaddq_f32(multiplier, ones);
      multiplier = vmulq_n_f32(multiplier, rms);
      float32x4_t tmp_Y = vmulq_f32(tmp_x, multiplier);
      vst1q_f32(y_ptr + d, tmp_Y);
    }
    for (; d < D; ++d) {
      float tmp_X = x_ptr[d];
      float multiplier = w_ptr[d] + 1.f;
      y_ptr[d] = tmp_X * rms * multiplier;
    }
  } else {
    int d;
    for (d = 0; d <= D - 4; ++d) {
      float32x4_t tmp_x = vld1q_f32(x_ptr + d);
      float32x4_t multiplier = vld1q_f32(w_ptr + d);
      multiplier = vmulq_n_f32(multiplier, rms);
      float32x4_t tmp_Y = vmulq_f32(tmp_x, multiplier);
      vst1q_f32(y_ptr + d, tmp_Y);
    }
    for (; d < D; ++d) {
      float tmp_X = x_ptr[d];
      float multiplier = w_ptr[d];
      y_ptr[d] = tmp_X * rms * multiplier;
    }
  }
}

// Should support [B, S, H * D] and [B, S, H, D]
void rmsnorm_fp16(const mllm_fp16_t* __restrict X, const mllm_fp16_t* __restrict W, mllm_fp16_t* __restrict Y, int D,
                  float epsilon, bool add_unit_offset, int thread_count) {
  auto x_ptr = X;
  auto y_ptr = Y;
  auto w_ptr = W;

  // pass 1: compute RMS scaling factor
  float mean_square = vsquare_mean_fp16(x_ptr, D);
  const float rms_float = 1.f / std::sqrtf(mean_square + epsilon);
  float16_t rms_fp16 = static_cast<float16_t>(rms_float);
  float16x8_t rms_vec = vdupq_n_f16(rms_fp16);

  // pass 2: apply scaling with weight
  if (add_unit_offset) {
    float16x8_t ones = vdupq_n_f16(1.0f);
    int d = 0;
    for (; d <= D - 8; d += 8) {
      float16x8_t tmp_x = vld1q_f16(x_ptr + d);
      float16x8_t multiplier = vld1q_f16(w_ptr + d);
      multiplier = vaddq_f16(multiplier, ones);
      multiplier = vmulq_f16(multiplier, rms_vec);
      float16x8_t tmp_y = vmulq_f16(tmp_x, multiplier);
      vst1q_f16(y_ptr + d, tmp_y);
    }
    for (; d < D; ++d) { y_ptr[d] = x_ptr[d] * rms_fp16 * (w_ptr[d] + 1.0f); }
  } else {
    int d = 0;
    for (; d <= D - 8; d += 8) {
      float16x8_t tmp_x = vld1q_f16(x_ptr + d);
      float16x8_t multiplier = vld1q_f16(w_ptr + d);
      multiplier = vmulq_f16(multiplier, rms_vec);
      float16x8_t tmp_y = vmulq_f16(tmp_x, multiplier);
      vst1q_f16(y_ptr + d, tmp_y);
    }
    for (; d < D; ++d) { y_ptr[d] = x_ptr[d] * rms_fp16 * w_ptr[d]; }
  }
}

}  // namespace mllm::cpu::arm

#endif