// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <cmath>
#include <arm_neon.h>

namespace mllm::cpu::arm {
void layernorm2d_fp32(const float* x, const float* weight, const float* bias, float* y, int N, int C, int H, int W, float eps) {
  const int spatial_dim = H * W;

  for (int n = 0; n < N; ++n) {
    for (int i = 0; i < spatial_dim; ++i) {
      const float* x_ptr = x + n * C * spatial_dim + i;
      float* y_ptr = y + n * C * spatial_dim + i;

      float sum = 0.0f;
#if defined(__ARM_NEON)
      float32x4_t sum_vec = vdupq_n_f32(0.0f);
      int c = 0;
      for (; c <= C - 4; c += 4) {
        float32x4_t x_vec = {x_ptr[c * spatial_dim], x_ptr[(c + 1) * spatial_dim], x_ptr[(c + 2) * spatial_dim],
                             x_ptr[(c + 3) * spatial_dim]};
        sum_vec = vaddq_f32(sum_vec, x_vec);
      }
      sum = vaddvq_f32(sum_vec);
      for (; c < C; ++c) { sum += x_ptr[c * spatial_dim]; }
#else
      for (int c = 0; c < C; ++c) { sum += x_ptr[c * spatial_dim]; }
#endif
      const float mean = sum / C;

      float sq_sum = 0.0f;
#if defined(__ARM_NEON)
      float32x4_t sq_sum_vec = vdupq_n_f32(0.0f);
      float32x4_t mean_vec = vdupq_n_f32(mean);
      c = 0;
      for (; c <= C - 4; c += 4) {
        float32x4_t x_vec = {x_ptr[c * spatial_dim], x_ptr[(c + 1) * spatial_dim], x_ptr[(c + 2) * spatial_dim],
                             x_ptr[(c + 3) * spatial_dim]};
        float32x4_t diff = vsubq_f32(x_vec, mean_vec);
        sq_sum_vec = vmlaq_f32(sq_sum_vec, diff, diff);  // Fused multiply-accumulate: sq_sum_vec += diff * diff
      }
      sq_sum = vaddvq_f32(sq_sum_vec);
      for (; c < C; ++c) {
        float diff = x_ptr[c * spatial_dim] - mean;
        sq_sum += diff * diff;
      }
#else
      for (int c = 0; c < C; ++c) {
        float diff = x_ptr[c * spatial_dim] - mean;
        sq_sum += diff * diff;
      }
#endif
      const float variance = sq_sum / C;
      const float inv_std = 1.0f / std::sqrt(variance + eps);

#if defined(__ARM_NEON)
      float32x4_t inv_std_vec = vdupq_n_f32(inv_std);
      c = 0;
      for (; c <= C - 4; c += 4) {
        float32x4_t x_vec = {x_ptr[c * spatial_dim], x_ptr[(c + 1) * spatial_dim], x_ptr[(c + 2) * spatial_dim],
                             x_ptr[(c + 3) * spatial_dim]};
        float32x4_t weight_vec = vld1q_f32(weight + c);
        float32x4_t bias_vec = vld1q_f32(bias + c);

        // y = (x - mean) * inv_std
        float32x4_t norm_val = vmulq_f32(vsubq_f32(x_vec, mean_vec), inv_std_vec);
        // y = y * weight + bias
        float32x4_t out_vec = vmlaq_f32(bias_vec, norm_val, weight_vec);

        y_ptr[c * spatial_dim] = vgetq_lane_f32(out_vec, 0);
        y_ptr[(c + 1) * spatial_dim] = vgetq_lane_f32(out_vec, 1);
        y_ptr[(c + 2) * spatial_dim] = vgetq_lane_f32(out_vec, 2);
        y_ptr[(c + 3) * spatial_dim] = vgetq_lane_f32(out_vec, 3);
      }
      for (; c < C; ++c) { y_ptr[c * spatial_dim] = (x_ptr[c * spatial_dim] - mean) * inv_std * weight[c] + bias[c]; }
#else
      for (int c = 0; c < C; ++c) { y_ptr[c * spatial_dim] = (x_ptr[c * spatial_dim] - mean) * inv_std * weight[c] + bias[c]; }
#endif
    }
  }
}

}  // namespace mllm::cpu::arm

#endif
