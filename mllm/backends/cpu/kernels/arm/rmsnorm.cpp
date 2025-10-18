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

void rmsnorm_fp32_inplace(const mllm_fp32_t* __restrict X, const mllm_fp32_t* __restrict W, mllm_fp32_t* __restrict Y, int D,
                          float epsilon, bool add_unit_offset, int thread_count) {
  (void)thread_count;
  const float* __restrict x = X;
  float* __restrict y = Y;
  const float* __restrict w = W;

  /* ---------- pass 1 : rms ---------- */
  float sum = 0.f;
  int d;
  for (d = 0; d <= D - 4; d += 4) {
    float32x4_t vx = vld1q_f32(x + d);
    sum += vaddvq_f32(vmulq_f32(vx, vx));
  }
  for (; d < D; ++d) sum += x[d] * x[d];
  const float rms = 1.f / std::sqrt(sum / D + epsilon);

  /* ---------- pass 2 : y = x * rms ---------- */
  for (d = 0; d <= D - 4; d += 4) {
    float32x4_t vx = vld1q_f32(x + d);
    vst1q_f32(y + d, vmulq_n_f32(vx, rms));
  }
  for (; d < D; ++d) y[d] = x[d] * rms;

  /* ---------- pass 3 : y = y * (w or w+1) ---------- */
  if (add_unit_offset) {
    const float32x4_t ones = vdupq_n_f32(1.f);
    for (d = 0; d <= D - 4; d += 4) {
      float32x4_t vy = vld1q_f32(y + d);
      float32x4_t vw = vld1q_f32(w + d);
      vw = vaddq_f32(vw, ones);
      vst1q_f32(y + d, vmulq_f32(vy, vw));
    }
    for (; d < D; ++d) y[d] *= (w[d] + 1.f);
  } else {
    for (d = 0; d <= D - 4; d += 4) {
      float32x4_t vy = vld1q_f32(y + d);
      float32x4_t vw = vld1q_f32(w + d);
      vst1q_f32(y + d, vmulq_f32(vy, vw));
    }
    for (; d < D; ++d) y[d] *= w[d];
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

void rmsnorm_fp16_inplace(const mllm_fp16_t* __restrict X, const mllm_fp16_t* __restrict W, mllm_fp16_t* __restrict Y, int D,
                          float epsilon, bool add_unit_offset, int thread_count) {
  (void)thread_count;
  const auto* x = reinterpret_cast<const float16_t*>(X);
  const auto* w = reinterpret_cast<const float16_t*>(W);
  auto* y = reinterpret_cast<float16_t*>(Y);

  /* ---------- pass 1 : rms (fp32 precision) ---------- */
  float sum = 0.f;
  int d = 0;
  for (; d <= D - 8; d += 8) {
    float16x8_t vx = vld1q_f16(x + d);
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(vx));
    float32x4_t hi = vcvt_f32_f16(vget_high_f16(vx));
    sum += vaddvq_f32(vmulq_f32(lo, lo));
    sum += vaddvq_f32(vmulq_f32(hi, hi));
  }
  for (; d < D; ++d) {
    float vx_f32 = static_cast<float>(x[d]);
    sum += vx_f32 * vx_f32;
  }
  const float rms_f32 = 1.f / std::sqrt(sum / D + epsilon);
  const float16_t rms_f16 = static_cast<float16_t>(rms_f32);
  const float16x8_t rms_vec = vdupq_n_f16(rms_f16);

  /* ---------- pass 2 : y = x * rms ---------- */
  for (d = 0; d <= D - 8; d += 8) {
    float16x8_t vx = vld1q_f16(x + d);
    vst1q_f16(y + d, vmulq_f16(vx, rms_vec));
  }
  for (; d < D; ++d) y[d] = x[d] * rms_f16;

  /* ---------- pass 3 : y = y * (w or w+1) ---------- */
  if (add_unit_offset) {
    const float16x8_t ones = vdupq_n_f16(1.0f);
    for (d = 0; d <= D - 8; d += 8) {
      float16x8_t vy = vld1q_f16(y + d);
      float16x8_t vw = vld1q_f16(w + d);
      vw = vaddq_f16(vw, ones);
      vst1q_f16(y + d, vmulq_f16(vy, vw));
    }
    for (; d < D; ++d) y[d] *= (w[d] + float16_t(1.0f));
  } else {
    for (d = 0; d <= D - 8; d += 8) {
      float16x8_t vy = vld1q_f16(y + d);
      float16x8_t vw = vld1q_f16(w + d);
      vst1q_f16(y + d, vmulq_f16(vy, vw));
    }
    for (; d < D; ++d) y[d] *= w[d];
  }
}

}  // namespace mllm::cpu::arm

#endif
