// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>

#include "mllm/utils/Common.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/backends/cpu/ops/VisionRoPEOp.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
// Include AVX, SSE.
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <arm_neon.h>
#endif

namespace mllm::cpu {

void Qwen2VLVisionRoPEOpImpl::forward(const Tensor& activation, const Tensor& sin, const Tensor& cos, Tensor& out) {
  // [B, S, H, D]
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  switch (activation.dtype()) {
    case kFloat16: {
      auto B = activation.shape()[0];
      auto S = activation.shape()[1];
      auto H = activation.shape()[2];
      auto D = activation.shape()[3];
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("Qwen2VLVisionRoPEOpImpl is not implemented for x86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      auto activation_ptr = activation.ptr<float16_t>();
      auto output_ptr = out.ptr<float16_t>();
      auto sin_ptr = sin.ptr<float16_t>();
      auto cos_ptr = cos.ptr<float16_t>();

      auto half_dim = D / 2;

      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
          for (int h = 0; h < H; ++h) {
            auto act_base = activation_ptr + b * S * H * D + s * H * D + h * D;
            auto out_base = output_ptr + b * S * H * D + s * H * D + h * D;

            auto sin_base = sin_ptr + s * half_dim;  // sin shape is [S, half_dim]
            auto cos_base = cos_ptr + s * half_dim;  // cos shape is [S, half_dim]

            int d = 0;
            for (; d + 3 < half_dim; d += 4) {
              float16x4_t a = vld1_f16(act_base + d);
              float16x4_t b = vld1_f16(act_base + d + half_dim);

              float16x4_t cos_val = vld1_f16(cos_base + d);
              float16x4_t sin_val = vld1_f16(sin_base + d);

              // part1 = a * cos_val - b * sin_val
              // part2 = a * sin_val + b * cos_val
              float16x4_t part1 = vsub_f16(vmul_f16(a, cos_val), vmul_f16(b, sin_val));
              float16x4_t part2 = vadd_f16(vmul_f16(a, sin_val), vmul_f16(b, cos_val));

              vst1_f16(out_base + d, part1);
              vst1_f16(out_base + d + half_dim, part2);
            }

            for (; d < half_dim; ++d) {
              const auto a = act_base[d];
              const auto b = act_base[d + half_dim];
              const auto cos_val = cos_base[d];
              const auto sin_val = sin_base[d];
              out_base[d] = a * cos_val - b * sin_val;
              out_base[d + half_dim] = a * sin_val + b * cos_val;
            }
          }
        }
      }
#endif
      break;
    }
    case kFloat32: {
      const auto B = activation.shape()[0];
      const auto S = activation.shape()[1];
      const auto H = activation.shape()[2];
      const auto D = activation.shape()[3];
      const auto half_dim = D / 2;

      auto activation_ptr = activation.ptr<float>();
      auto output_ptr = out.ptr<float>();
      auto sin_ptr = sin.ptr<float>();
      auto cos_ptr = cos.ptr<float>();

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("Qwen2VLVisionRoPEOpImpl is not implemented for x86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
          for (int h = 0; h < H; ++h) {
            auto act_base = activation_ptr + b * S * H * D + s * H * D + h * D;
            auto out_base = output_ptr + b * S * H * D + s * H * D + h * D;

            auto sin_base = sin_ptr + s * half_dim;  // sin shape is [S, half_dim]
            auto cos_base = cos_ptr + s * half_dim;  // cos shape is [S, half_dim]

            int d = 0;
            for (; d + 3 < half_dim; d += 4) {
              float32x4_t a_front = vld1q_f32(act_base + d);
              float32x4_t a_back = vld1q_f32(act_base + d + half_dim);

              float32x4_t cos_val = vld1q_f32(cos_base + d);
              float32x4_t sin_val = vld1q_f32(sin_base + d);

              // out_front = a_front * cos_val - a_back * sin_val
              // out_back  = a_front * sin_val + a_back * cos_val
              float32x4_t out_front = vmlsq_f32(vmulq_f32(a_front, cos_val), a_back, sin_val);
              float32x4_t out_back = vmlaq_f32(vmulq_f32(a_front, sin_val), a_back, cos_val);

              vst1q_f32(out_base + d, out_front);
              vst1q_f32(out_base + d + half_dim, out_back);
            }

            for (; d < half_dim; ++d) {
              const float a_front = act_base[d];
              const float a_back = act_base[d + half_dim];
              const float cos_val = cos_base[d];
              const float sin_val = sin_base[d];
              out_base[d] = a_front * cos_val - a_back * sin_val;
              out_base[d + half_dim] = a_front * sin_val + a_back * cos_val;
            }
          }
        }
      }
#endif
      break;
    }
    default: {
      NYI("Unsupported activation type");
    }
  }
}

CPUVisionRoPEOp::CPUVisionRoPEOp(const aops::VisionRoPEOpOptions& options) : aops::VisionRoPEOp(options) {}

void CPUVisionRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& activation = inputs[0];

  // Only support BSHD inputs.
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  // Input is BSHD, seq_len is in pos 1.
  auto seq_len = activation.shape()[1];

  switch (options_.type) {
    case aops::VisionRoPEOpOptionsType::kQwen2VL: {
      // For Qwen2VL's ViT RoPE
      auto impl = Qwen2VLVisionRoPEOpImpl();

      Tensor sin = Tensor::nil();
      Tensor cos = Tensor::nil();

      // Means we have sin and cos
      if (inputs.size() > 2) {
        sin = inputs[1];
        cos = inputs[2];
      }

      // Compute sin and cos.
      if (!sin && !cos) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "VisionRoPEOp: sin and cos are nil."); }

      // Do VisionRoPE Operation.
      impl.forward(activation, sin, cos, outputs[0]);
      break;
    }
    default: {
      NYI("Unsupported VisionRoPEOpOptionsType");
      break;
    }
  }
}

}  // namespace mllm::cpu
