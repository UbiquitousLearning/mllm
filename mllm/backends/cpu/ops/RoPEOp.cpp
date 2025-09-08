// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/RoPEOp.hpp"

#include "mllm/utils/CPUArchHelper.hpp"
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
// Include AVX, SSE.
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <arm_neon.h>
#endif

namespace mllm::cpu {

void RoPEOpImpl::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Tensor& sin, Tensor& cos) {
  auto activation = inputs[0];
  auto out = outputs[0];

  // Activation must in BHSD layout
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  auto B = activation.shape()[0];
  auto H = activation.shape()[1];
  auto S = activation.shape()[2];
  auto D = activation.shape()[3];

  int32_t half = D / 2;

  switch (activation.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      for (int n = 0; n < B; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            float* act_ptr = activation.offsettedPtr<float>({n, h, s, 0});
            float* out_ptr = out.offsettedPtr<float>({n, h, s, 0});
            const float* sin_ptr = sin.offsettedPtr<float>({n, s, 0});
            const float* cos_ptr = cos.offsettedPtr<float>({n, s, 0});

            for (int d = 0; d < half; ++d) {
              float in_val = act_ptr[d];
              float in_val2 = act_ptr[d + half];
              float sin_val = sin_ptr[d];
              float cos_val = cos_ptr[d];

              out_ptr[d] = in_val * cos_val - in_val2 * sin_val;
              out_ptr[d + half] = in_val * sin_val + in_val2 * cos_val;
            }
          }
        }
      }
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      for (int n = 0; n < B; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            float* act_ptr = activation.offsettedPtr<float>({n, h, s, 0});
            float* out_ptr = out.offsettedPtr<float>({n, h, s, 0});
            const float* sin_ptr = sin.offsettedPtr<float>({n, s, 0});
            const float* cos_ptr = cos.offsettedPtr<float>({n, s, 0});

            // Vectorized processing (4 elements per iteration)
            int d = 0;
            constexpr int step = 4;
            for (; d <= half - step; d += step) {
              // Load activation blocks
              float32x4_t act_front = vld1q_f32(act_ptr + d);
              float32x4_t act_back = vld1q_f32(act_ptr + d + half);

              // Load sin/cos values
              float32x4_t sin_vec = vld1q_f32(sin_ptr + d);
              float32x4_t cos_vec = vld1q_f32(cos_ptr + d);

              // Compute rotated values
              float32x4_t out_front = vsubq_f32(vmulq_f32(act_front, cos_vec), vmulq_f32(act_back, sin_vec));
              float32x4_t out_back = vaddq_f32(vmulq_f32(act_front, sin_vec), vmulq_f32(act_back, cos_vec));

              // Store results
              vst1q_f32(out_ptr + d, out_front);
              vst1q_f32(out_ptr + d + half, out_back);
            }

            // Process remaining elements
            for (; d < half; ++d) {
              float in_val = act_ptr[d];
              float in_val2 = act_ptr[d + half];
              float sin_val = sin_ptr[d];
              float cos_val = cos_ptr[d];

              out_ptr[d] = in_val * cos_val - in_val2 * sin_val;
              out_ptr[d + half] = in_val * sin_val + in_val2 * cos_val;
            }
          }
        }
      }
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      for (int n = 0; n < B; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            float16_t* act_ptr = activation.offsettedPtr<float16_t>({n, h, s, 0});
            float16_t* out_ptr = out.offsettedPtr<float16_t>({n, h, s, 0});
            const float16_t* sin_ptr = sin.offsettedPtr<float16_t>({n, s, 0});
            const float16_t* cos_ptr = cos.offsettedPtr<float16_t>({n, s, 0});

            for (int d = 0; d < half; ++d) {
              float in_val = static_cast<float>(act_ptr[d]);
              float in_val2 = static_cast<float>(act_ptr[d + half]);
              float sin_val = static_cast<float>(sin_ptr[d]);
              float cos_val = static_cast<float>(cos_ptr[d]);

              out_ptr[d] = static_cast<float16_t>(in_val * cos_val - in_val2 * sin_val);
              out_ptr[d + half] = static_cast<float16_t>(in_val * sin_val + in_val2 * cos_val);
            }
          }
        }
      }
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      for (int n = 0; n < B; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            float16_t* act_ptr = activation.offsettedPtr<float16_t>({n, h, s, 0});
            float16_t* out_ptr = out.offsettedPtr<float16_t>({n, h, s, 0});
            const float16_t* sin_ptr = sin.offsettedPtr<float16_t>({n, s, 0});
            const float16_t* cos_ptr = cos.offsettedPtr<float16_t>({n, s, 0});

            // Vectorized processing (8 elements per iteration)
            int d = 0;
            constexpr int step = 8;
            for (; d <= half - step; d += step) {
              // Load activation blocks
              float16x8_t act_front = vld1q_f16(act_ptr + d);
              float16x8_t act_back = vld1q_f16(act_ptr + d + half);

              // Load sin/cos values
              float16x8_t sin_vec = vld1q_f16(sin_ptr + d);
              float16x8_t cos_vec = vld1q_f16(cos_ptr + d);

              // Compute rotated values
              float16x8_t out_front = vsubq_f16(vmulq_f16(act_front, cos_vec), vmulq_f16(act_back, sin_vec));
              float16x8_t out_back = vaddq_f16(vmulq_f16(act_front, sin_vec), vmulq_f16(act_back, cos_vec));

              // Store results
              vst1q_f16(out_ptr + d, out_front);
              vst1q_f16(out_ptr + d + half, out_back);
            }

            // Process remaining elements
            for (; d < half; ++d) {
              float in_val = static_cast<float>(act_ptr[d]);
              float in_val2 = static_cast<float>(act_ptr[d + half]);
              float sin_val = static_cast<float>(sin_ptr[d]);
              float cos_val = static_cast<float>(cos_ptr[d]);

              out_ptr[d] = static_cast<float16_t>(in_val * cos_val - in_val2 * sin_val);
              out_ptr[d + half] = static_cast<float16_t>(in_val * sin_val + in_val2 * cos_val);
            }
          }
        }
      }
#endif
      break;
    }
    default: {
      NYI("RoPEOpImpl::forward not support this dtype")
      break;
    }
  }
}

CPURoPEOp::CPURoPEOp(const aops::RoPEOpOptions& options) : aops::RoPEOp(options) {}

void CPURoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Expect 3 inputs:
  // Pos 0: activations
  // Pos 1: sin
  // Pos 2: cos
  MLLM_RT_ASSERT_EQ(inputs.size(), 3);

  auto& activation = inputs[0];
  auto sin = inputs[1];
  auto cos = inputs[2];

  // Input must be [B, H, S, D]
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);
  auto out = outputs[0];

  auto impl = RoPEOpImpl();
  impl.forward(inputs, outputs, sin, cos);
}

}  // namespace mllm::cpu