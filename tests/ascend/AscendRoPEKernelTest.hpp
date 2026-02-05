// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Functional.hpp"
#include "KernelTestHelper.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"
#include <vector>
#include <cmath>

class AscendRoPEKernelTest : public KernelTest {
 public:
  AscendRoPEKernelTest() = default;
  ~AscendRoPEKernelTest() override = default;

  // Test RoPE operation with different shapes
  // Shape format: {B, H, S, D} where D must be even (split into two halves for rotation)
  bool RoPEFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      MLLM_RT_ASSERT(shape.size() == 4);
      int B = shape[0];
      int H = shape[1];
      int S = shape[2];
      int D = shape[3];
      MLLM_RT_ASSERT(D % 2 == 0);

      int half_D = D / 2;

      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -2.0f, 2.0f, kFloat16, kCPU);

      // 2. Generate cos and sin tables
      // Shape: [1, S, 1, D] to broadcast correctly
      Tensor cos_cpu = Tensor::zeros({1, S, 1, D}, kFloat16, kCPU);
      Tensor sin_cpu = Tensor::zeros({1, S, 1, D}, kFloat16, kCPU);

      {
        auto* cos_ptr = cos_cpu.ptr<mllm_fp16_t>();
        auto* sin_ptr = sin_cpu.ptr<mllm_fp16_t>();

        // Generate RoPE frequencies
        // freq_i = 1 / (theta^(2i/D)) where theta = 10000
        float theta = 10000.0f;
        for (int s = 0; s < S; ++s) {
          for (int d = 0; d < half_D; ++d) {
            float freq = 1.0f / std::pow(theta, (2.0f * d) / D);
            float angle = s * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            // Store cos/sin for both halves (interleaved or split depending on implementation)
            // ATB RoPE with rotaryCoeff=2 expects: [cos, cos] pattern for D dimension
            int idx = s * D + d;
            int idx2 = s * D + d + half_D;
            cos_ptr[idx] = MLLM_FP32_TO_FP16(cos_val);
            cos_ptr[idx2] = MLLM_FP32_TO_FP16(cos_val);
            sin_ptr[idx] = MLLM_FP32_TO_FP16(sin_val);
            sin_ptr[idx2] = MLLM_FP32_TO_FP16(sin_val);
          }
        }
      }

      // 3. Compute reference result (FP16) on CPU
      // RoPE formula (half rotation, rotaryCoeff=2):
      // For each position, split x into [x1, x2] (first half, second half)
      // x1_new = x1 * cos - x2 * sin
      // x2_new = x1 * sin + x2 * cos
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* cos_ptr = cos_cpu.ptr<mllm_fp16_t>();
        auto* sin_ptr = sin_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();

        for (int b = 0; b < B; ++b) {
          for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
              for (int d = 0; d < half_D; ++d) {
                // Index into x: [b, h, s, d] and [b, h, s, d + half_D]
                int idx1 = ((b * H + h) * S + s) * D + d;
                int idx2 = ((b * H + h) * S + s) * D + d + half_D;

                // Index into cos/sin: [0, s, 0, d]
                int cs_idx = s * D + d;

                float x1 = MLLM_FP16_TO_FP32(x_ptr[idx1]);
                float x2 = MLLM_FP16_TO_FP32(x_ptr[idx2]);
                float cos_val = MLLM_FP16_TO_FP32(cos_ptr[cs_idx]);
                float sin_val = MLLM_FP16_TO_FP32(sin_ptr[cs_idx]);

                // Apply rotation
                float x1_new = x1 * cos_val - x2 * sin_val;
                float x2_new = x1 * sin_val + x2 * cos_val;

                r_ptr[idx1] = MLLM_FP32_TO_FP16(x1_new);
                r_ptr[idx2] = MLLM_FP32_TO_FP16(x2_new);
              }
            }
          }
        }
      }

      // 4. Move inputs to Ascend and run RoPE
      // MLLM format: [B, H, S, D]
      // ATB RoPE expects: [B, S, H, D]
      // So we need to transpose before and after RoPE
      auto x_ascend = x_cpu.to(kAscend);
      auto cos_ascend = cos_cpu.to(kAscend);
      auto sin_ascend = sin_cpu.to(kAscend);

      // Transpose [B, H, S, D] -> [B, S, H, D] before RoPE
      auto x_transposed = x_ascend.transpose(1, 2);

      auto y_transposed = mllm::nn::functional::rope(x_transposed, cos_ascend, sin_ascend);

      // Transpose [B, S, H, D] -> [B, H, S, D] after RoPE
      auto y_ascend = y_transposed.transpose(1, 2);

      // 5. Move result back to CPU and compare with reference using allClose
      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        MLLM_ERROR("RoPE test failed for shape [{}, {}, {}, {}]", B, H, S, D);

        // Debug: print first few mismatched values
        auto* y_ptr = y_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        int mismatch_count = 0;
        const int max_print = 10;

        std::cout << "\n=== Debug: First " << max_print << " mismatches ===" << std::endl;
        for (int b = 0; b < B && mismatch_count < max_print; ++b) {
          for (int h = 0; h < H && mismatch_count < max_print; ++h) {
            for (int s = 0; s < S && mismatch_count < max_print; ++s) {
              for (int d = 0; d < D && mismatch_count < max_print; ++d) {
                int idx = ((b * H + h) * S + s) * D + d;
                float actual = MLLM_FP16_TO_FP32(y_ptr[idx]);
                float expected = MLLM_FP16_TO_FP32(r_ptr[idx]);
                float input = MLLM_FP16_TO_FP32(x_ptr[idx]);
                float diff = std::abs(actual - expected);
                if (diff > 1e-2f) {
                  std::cout << "  [b=" << b << ",h=" << h << ",s=" << s << ",d=" << d << "] "
                            << "input=" << input << ", actual=" << actual
                            << ", expected=" << expected << ", diff=" << diff << std::endl;
                  mismatch_count++;
                }
              }
            }
          }
        }
        std::cout << "=== End Debug ===" << std::endl;

        return false;
      }
      MLLM_INFO("RoPE test passed for shape [{}, {}, {}, {}]", B, H, S, D);
    }
    return true;
  }
};
