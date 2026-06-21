// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/core/Parallel.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include "mllm/backends/cpu/kernels/arm/math.hpp"
#include "mllm/backends/cpu/kernels/arm/gelu.hpp"

namespace mllm::cpu::arm {

void gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N, int thread_cnt) {
  const float32x4_t alpha = vdupq_n_f32(0.044715f);
  const float32x4_t beta = vdupq_n_f32(0.79788456f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t half = vdupq_n_f32(0.5f);

  if (thread_cnt > 1) {
    int tails = N % 4;
    int loops = N < 4 ? 0 : N - tails;
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, loops, 4, thread_cnt) {
      float32x4_t x = vld1q_f32(X + i);

      float32x4_t x3 = vmulq_f32(x, vmulq_f32(x, x));

      float32x4_t inner = vmlaq_f32(x, alpha, x3);

      float32x4_t scaled = vmulq_f32(beta, inner);

      float32x4_t tanh_val = vtanh_fp32(scaled);

      float32x4_t result = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_val));
      vst1q_f32(Z + i, result);
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()
    for (int i = loops; i < N; i++) {
      float x = X[i];
      float x3 = x * x * x;
      float inner = x + 0.044715f * x3;
      float scaled = 0.79788456f * inner;
      float tanh_val;
      {
        float32x4_t tmp = vtanh_fp32(vdupq_n_f32(scaled));
        tanh_val = vgetq_lane_f32(tmp, 0);
      }
      Z[i] = 0.5f * x * (1.0f + tanh_val);
    }
  } else {
    int i = 0;
    for (i = 0; i <= N - 4; i += 4) {
      float32x4_t x = vld1q_f32(X + i);

      float32x4_t x3 = vmulq_f32(x, vmulq_f32(x, x));

      float32x4_t inner = vmlaq_f32(x, alpha, x3);

      float32x4_t scaled = vmulq_f32(beta, inner);

      float32x4_t tanh_val = vtanh_fp32(scaled);

      float32x4_t result = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_val));
      vst1q_f32(Z + i, result);
    }

    for (; i < N; i++) {
      float x = X[i];
      float x3 = x * x * x;
      float inner = x + 0.044715f * x3;
      float scaled = 0.79788456f * inner;
      float tanh_val;
      {
        float32x4_t tmp = vtanh_fp32(vdupq_n_f32(scaled));
        tanh_val = vgetq_lane_f32(tmp, 0);
      }
      Z[i] = 0.5f * x * (1.0f + tanh_val);
    }
  }
}

void quick_gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N, int thread_cnt) {
  const float32x4_t scale = vdupq_n_f32(1.702f);
  const float32x4_t one = vdupq_n_f32(1.0f);

  if (thread_cnt > 1) {
    int tails = N % 4;
    int loops = N < 4 ? 0 : N - tails;
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, loops, 4, thread_cnt) {
      float32x4_t x = vld1q_f32(X + i);
      float32x4_t scaled_x = vmulq_f32(x, scale);
      float32x4_t sigmoid_val = vsigmoid_f32(scaled_x);
      float32x4_t result = vmulq_f32(x, sigmoid_val);
      vst1q_f32(Z + i, result);
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()
    for (int i = loops; i < N; i++) {
      float x = X[i];
      float scaled_x = 1.702f * x;
      float sigmoid_val = 1.0f / (1.0f + expf(-scaled_x));
      Z[i] = x * sigmoid_val;
    }
  } else {
    int i = 0;
    for (i = 0; i <= N - 4; i += 4) {
      float32x4_t x = vld1q_f32(X + i);
      float32x4_t scaled_x = vmulq_f32(x, scale);
      float32x4_t sigmoid_val = vsigmoid_f32(scaled_x);
      float32x4_t result = vmulq_f32(x, sigmoid_val);
      vst1q_f32(Z + i, result);
    }

    for (; i < N; i++) {
      float x = X[i];
      float scaled_x = 1.702f * x;
      float sigmoid_val = 1.0f / (1.0f + expf(-scaled_x));
      Z[i] = x * sigmoid_val;
    }
  }
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void gelu_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X, int32_t N, int thread_cnt) {
  const float32x4_t alpha = vdupq_n_f32(0.044715f);
  const float32x4_t beta = vdupq_n_f32(0.79788456f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t half = vdupq_n_f32(0.5f);

  if (thread_cnt > 1) {
    int tails = N % 8;
    int loops = N < 8 ? 0 : N - tails;
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, loops, 8, thread_cnt) {
      float16x8_t x_half = vld1q_f16(X + i);

      float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_half));
      float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_half));

      float32x4_t x3_low = vmulq_f32(x_low, vmulq_f32(x_low, x_low));
      float32x4_t inner_low = vmlaq_f32(x_low, alpha, x3_low);
      float32x4_t scaled_low = vmulq_f32(beta, inner_low);
      float32x4_t tanh_low = vtanh_fp32(scaled_low);
      float32x4_t result_low = vmulq_f32(vmulq_f32(half, x_low), vaddq_f32(one, tanh_low));

      float32x4_t x3_high = vmulq_f32(x_high, vmulq_f32(x_high, x_high));
      float32x4_t inner_high = vmlaq_f32(x_high, alpha, x3_high);
      float32x4_t scaled_high = vmulq_f32(beta, inner_high);
      float32x4_t tanh_high = vtanh_fp32(scaled_high);
      float32x4_t result_high = vmulq_f32(vmulq_f32(half, x_high), vaddq_f32(one, tanh_high));

      float16x4_t res_low_half = vcvt_f16_f32(result_low);
      float16x4_t res_high_half = vcvt_f16_f32(result_high);
      vst1q_f16(Z + i, vcombine_f16(res_low_half, res_high_half));
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()
    for (int i = loops; i < N; i++) {
      float x = (float)X[i];
      float x3 = x * x * x;
      float inner = x + 0.044715f * x3;
      float scaled = 0.79788456f * inner;

      float tanh_val;
      float32x4_t tmp = vtanh_fp32(vdupq_n_f32(scaled));
      tanh_val = vgetq_lane_f32(tmp, 0);

      Z[i] = (__fp16)(0.5f * x * (1.0f + tanh_val));
    }
  } else {
    int i = 0;
    for (i = 0; i <= N - 8; i += 8) {
      float16x8_t x_half = vld1q_f16(X + i);

      float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_half));
      float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_half));

      float32x4_t x3_low = vmulq_f32(x_low, vmulq_f32(x_low, x_low));
      float32x4_t inner_low = vmlaq_f32(x_low, alpha, x3_low);
      float32x4_t scaled_low = vmulq_f32(beta, inner_low);
      float32x4_t tanh_low = vtanh_fp32(scaled_low);
      float32x4_t result_low = vmulq_f32(vmulq_f32(half, x_low), vaddq_f32(one, tanh_low));

      float32x4_t x3_high = vmulq_f32(x_high, vmulq_f32(x_high, x_high));
      float32x4_t inner_high = vmlaq_f32(x_high, alpha, x3_high);
      float32x4_t scaled_high = vmulq_f32(beta, inner_high);
      float32x4_t tanh_high = vtanh_fp32(scaled_high);
      float32x4_t result_high = vmulq_f32(vmulq_f32(half, x_high), vaddq_f32(one, tanh_high));

      float16x4_t res_low_half = vcvt_f16_f32(result_low);
      float16x4_t res_high_half = vcvt_f16_f32(result_high);
      vst1q_f16(Z + i, vcombine_f16(res_low_half, res_high_half));
    }

    for (; i < N; i++) {
      float x = (float)X[i];
      float x3 = x * x * x;
      float inner = x + 0.044715f * x3;
      float scaled = 0.79788456f * inner;

      float tanh_val;
      float32x4_t tmp = vtanh_fp32(vdupq_n_f32(scaled));
      tanh_val = vgetq_lane_f32(tmp, 0);

      Z[i] = (__fp16)(0.5f * x * (1.0f + tanh_val));
    }
  }
}

void quick_gelu_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X, int32_t N, int thread_cnt) {
  const float32x4_t scale = vdupq_n_f32(1.702f);
  const float32x4_t one = vdupq_n_f32(1.0f);

  if (thread_cnt > 1) {
    int tails = N % 8;
    int loops = N < 8 ? 0 : N - tails;
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, loops, 8, thread_cnt) {
      float16x8_t x_half = vld1q_f16(X + i);
      float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_half));
      float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_half));

      float32x4_t scaled_x_low = vmulq_f32(x_low, scale);
      float32x4_t scaled_x_high = vmulq_f32(x_high, scale);

      float32x4_t sigmoid_low = vsigmoid_f32(scaled_x_low);
      float32x4_t sigmoid_high = vsigmoid_f32(scaled_x_high);

      float32x4_t result_low = vmulq_f32(x_low, sigmoid_low);
      float32x4_t result_high = vmulq_f32(x_high, sigmoid_high);

      vst1q_f16(Z + i, vcombine_f16(vcvt_f16_f32(result_low), vcvt_f16_f32(result_high)));
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()
    for (int i = loops; i < N; i++) {
      float x = static_cast<float>(X[i]);
      float scaled_x = 1.702f * x;
      float sigmoid_val = 1.0f / (1.0f + expf(-scaled_x));
      Z[i] = static_cast<float16_t>(x * sigmoid_val);
    }
  } else {
    int i = 0;
    for (i = 0; i <= N - 8; i += 8) {
      float16x8_t x_half = vld1q_f16(X + i);
      float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_half));
      float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_half));

      float32x4_t scaled_x_low = vmulq_f32(x_low, scale);
      float32x4_t scaled_x_high = vmulq_f32(x_high, scale);

      float32x4_t sigmoid_low = vsigmoid_f32(scaled_x_low);
      float32x4_t sigmoid_high = vsigmoid_f32(scaled_x_high);

      float32x4_t result_low = vmulq_f32(x_low, sigmoid_low);
      float32x4_t result_high = vmulq_f32(x_high, sigmoid_high);

      vst1q_f16(Z + i, vcombine_f16(vcvt_f16_f32(result_low), vcvt_f16_f32(result_high)));
    }

    for (; i < N; i++) {
      float x = static_cast<float>(X[i]);
      float scaled_x = 1.702f * x;
      float sigmoid_val = 1.0f / (1.0f + expf(-scaled_x));
      Z[i] = static_cast<float16_t>(x * sigmoid_val);
    }
  }
}
#endif

}  // namespace mllm::cpu::arm

#endif
