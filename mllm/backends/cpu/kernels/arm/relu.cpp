// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/relu.hpp"
#include "mllm/core/Parallel.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::arm {

void relu_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int thread_count) {
  if (thread_count > 1) {
    int tails = len % 16;
    int _16_loops = len < 16 ? 0 : len - tails;
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, _16_loops, 16, thread_count) {
      float32x4_t x_line_0 = vld1q_f32(X + i);
      float32x4_t ans_line_0 = vmaxq_f32(x_line_0, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i, ans_line_0);

      float32x4_t x_line_1 = vld1q_f32(X + i + 4);
      float32x4_t ans_line_1 = vmaxq_f32(x_line_1, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i + 4, ans_line_1);

      float32x4_t x_line_2 = vld1q_f32(X + i + 8);
      float32x4_t ans_line_2 = vmaxq_f32(x_line_2, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i + 8, ans_line_2);

      float32x4_t x_line_3 = vld1q_f32(X + i + 12);
      float32x4_t ans_line_3 = vmaxq_f32(x_line_3, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i + 12, ans_line_3);
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()
    int i = _16_loops;
    for (; i <= len - 8; i += 8) {
      float32x4_t x_line_0 = vld1q_f32(X + i);
      float32x4_t ans_line_0 = vmaxq_f32(x_line_0, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i, ans_line_0);

      float32x4_t x_line_1 = vld1q_f32(X + i + 4);
      float32x4_t ans_line_1 = vmaxq_f32(x_line_1, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i + 4, ans_line_1);
    }
    for (; i <= len - 4; i += 4) {
      float32x4_t x_line_0 = vld1q_f32(X + i);
      float32x4_t ans_line_0 = vmaxq_f32(x_line_0, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i, ans_line_0);
    }
    for (; i < len; i++) { Y[i] = X[i] > 0.0f ? X[i] : 0.0f; }
  } else {
    int i;
    for (i = 0; i <= len - 16; i += 16) {
      float32x4_t x_line_0 = vld1q_f32(X + i);
      float32x4_t ans_line_0 = vmaxq_f32(x_line_0, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i, ans_line_0);

      float32x4_t x_line_1 = vld1q_f32(X + i + 4);
      float32x4_t ans_line_1 = vmaxq_f32(x_line_1, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i + 4, ans_line_1);

      float32x4_t x_line_2 = vld1q_f32(X + i + 8);
      float32x4_t ans_line_2 = vmaxq_f32(x_line_2, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i + 8, ans_line_2);

      float32x4_t x_line_3 = vld1q_f32(X + i + 12);
      float32x4_t ans_line_3 = vmaxq_f32(x_line_3, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i + 12, ans_line_3);
    }
    for (; i <= len - 8; i += 8) {
      float32x4_t x_line_0 = vld1q_f32(X + i);
      float32x4_t ans_line_0 = vmaxq_f32(x_line_0, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i, ans_line_0);

      float32x4_t x_line_1 = vld1q_f32(X + i + 4);
      float32x4_t ans_line_1 = vmaxq_f32(x_line_1, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i + 4, ans_line_1);
    }
    for (; i <= len - 4; i += 4) {
      float32x4_t x_line_0 = vld1q_f32(X + i);
      float32x4_t ans_line_0 = vmaxq_f32(x_line_0, vdupq_n_f32(0.0f));
      vst1q_f32(Y + i, ans_line_0);
    }
    for (; i < len; i++) { Y[i] = X[i] > 0.0f ? X[i] : 0.0f; }
  }
}

void relu_fp16(const mllm_fp16_t* __restrict X, mllm_fp16_t* __restrict Y, int len, int thread_count) {
  if (thread_count > 1) {
    int tails = len % 16;
    int _16_loops = len < 16 ? 0 : len - tails;
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, _16_loops, 16, thread_count) {
      float16x8_t x0 = vld1q_f16(X + i);
      float16x8_t relu0 = vmaxq_f16(x0, vdupq_n_f16(0.0f));
      vst1q_f16(Y + i, relu0);

      float16x8_t x1 = vld1q_f16(X + i + 8);
      float16x8_t relu1 = vmaxq_f16(x1, vdupq_n_f16(0.0f));
      vst1q_f16(Y + i + 8, relu1);
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()
    int i = _16_loops;
    for (; i <= len - 8; i += 8) {
      float16x8_t x = vld1q_f16(X + i);
      float16x8_t relu = vmaxq_f16(x, vdupq_n_f16(0.0f));
      vst1q_f16(Y + i, relu);
    }

    for (; i < len; ++i) { Y[i] = X[i] > 0.0f ? X[i] : 0.0f; }
  } else {
    int i = 0;
    for (i = 0; i <= len - 16; i += 16) {
      float16x8_t x0 = vld1q_f16(X + i);
      float16x8_t relu0 = vmaxq_f16(x0, vdupq_n_f16(0.0f));
      vst1q_f16(Y + i, relu0);

      float16x8_t x1 = vld1q_f16(X + i + 8);
      float16x8_t relu1 = vmaxq_f16(x1, vdupq_n_f16(0.0f));
      vst1q_f16(Y + i + 8, relu1);
    }
    for (; i <= len - 8; i += 8) {
      float16x8_t x = vld1q_f16(X + i);
      float16x8_t relu = vmaxq_f16(x, vdupq_n_f16(0.0f));
      vst1q_f16(Y + i, relu);
    }

    for (; i < len; ++i) { Y[i] = X[i] > 0.0f ? X[i] : 0.0f; }
  }
}

}  // namespace mllm::cpu::arm

#endif
