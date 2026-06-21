// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/MultimodalRoPEOp.hpp"

#include "mllm/core/aops/MultimodalRoPEOp.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
// Include AVX, SSE.
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <arm_neon.h>
#endif

namespace mllm::cpu {

void Qwen2VLMultimodalRoPEOpImpl::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Tensor& sin,
                                          Tensor& cos, aops::MultimodalRoPEOpOptionsInputType input_type) {
  auto activation = inputs[0];
  auto out = outputs[0];

  // Activation must in BHSD layout
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  int32_t B = 0;
  int32_t S = 0;
  int32_t H = 0;
  int32_t D = 0;

  switch (input_type) {
    case aops::MultimodalRoPEOpOptionsInputType::kBHSD: {
      B = activation.shape()[0];
      H = activation.shape()[1];
      S = activation.shape()[2];
      D = activation.shape()[3];
      break;
    }
    case aops::MultimodalRoPEOpOptionsInputType::kBSHD: {
      B = activation.shape()[0];
      S = activation.shape()[1];
      H = activation.shape()[2];
      D = activation.shape()[3];
      break;
    }
  }

  int32_t partial_dimension = D;
  int32_t half = D / 2;

  switch (activation.dtype()) {
    case kFloat32: {
      switch (input_type) {
        case aops::MultimodalRoPEOpOptionsInputType::kBHSD: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          NYI("Qwen2VLVisionRoPEOpImpl is not implemented for x86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                float* act_ptr = activation.offsettedPtr<float>({n, h, s, 0});
                float* out_ptr = out.offsettedPtr<float>({n, h, s, 0});
                const float* sin_ptr = sin.offsettedPtr<float>({s, 0});
                const float* cos_ptr = cos.offsettedPtr<float>({s, 0});

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
        case aops::MultimodalRoPEOpOptionsInputType::kBSHD: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          NYI("Qwen2VLVisionRoPEOpImpl is not implemented for x86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          for (int n = 0; n < B; ++n) {
            for (int s = 0; s < S; ++s) {
              for (int h = 0; h < H; ++h) {
                float* act_ptr = activation.offsettedPtr<float>({n, s, h, 0});
                float* out_ptr = out.offsettedPtr<float>({n, s, h, 0});
                const float* sin_ptr = sin.offsettedPtr<float>({s, 0});
                const float* cos_ptr = cos.offsettedPtr<float>({s, 0});

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
      }
      break;
    }
    case kFloat16: {
      switch (input_type) {
        case aops::MultimodalRoPEOpOptionsInputType::kBHSD: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          NYI("Qwen2VLVisionRoPEOpImpl is not implemented for x86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                float16_t* act_ptr = activation.offsettedPtr<float16_t>({n, h, s, 0});
                float16_t* out_ptr = out.offsettedPtr<float16_t>({n, h, s, 0});
                const float16_t* sin_ptr = sin.offsettedPtr<float16_t>({s, 0});
                const float16_t* cos_ptr = cos.offsettedPtr<float16_t>({s, 0});

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
        case aops::MultimodalRoPEOpOptionsInputType::kBSHD: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          NYI("Qwen2VLVisionRoPEOpImpl is not implemented for x86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          for (int n = 0; n < B; ++n) {
            for (int s = 0; s < S; ++s) {
              for (int h = 0; h < H; ++h) {
                float16_t* act_ptr = activation.offsettedPtr<float16_t>({n, s, h, 0});
                float16_t* out_ptr = out.offsettedPtr<float16_t>({n, s, h, 0});
                const float16_t* sin_ptr = sin.offsettedPtr<float16_t>({s, 0});
                const float16_t* cos_ptr = cos.offsettedPtr<float16_t>({s, 0});

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
      }
      break;
    }
    default: {
      NYI("Qwen2VLMultimodalRoPEOpImpl::forward not support this dtype")
      break;
    }
  }
}

CPUMultimodalRoPEOp::CPUMultimodalRoPEOp(const aops::MultimodalRoPEOpOptions& options) : aops::MultimodalRoPEOp(options) {}

void CPUMultimodalRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Expect 2 inputs:
  // Pos 1: activations
  // Pos 2: sin
  // Pos 2: cos
  MLLM_RT_ASSERT_EQ(inputs.size(), 3);

  auto& activation = inputs[0];
  auto sin = inputs[1];
  auto cos = inputs[2];

  // Input must be [B, H, S, D]
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);
  auto out = outputs[0];

  switch (options_.type) {
    case aops::MultimodalRoPEOpOptionsType::kQwen2VL: {
      auto impl = Qwen2VLMultimodalRoPEOpImpl();
      impl.forward(inputs, outputs, sin, cos, options_.input_type);
      break;
    }
    default: {
      NYI("Unsupported");
      break;
    }
  }
}

}  // namespace mllm::cpu
