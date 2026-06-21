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

class AscendSiLUKernelTest : public KernelTest {
 public:
  AscendSiLUKernelTest() = default;
  ~AscendSiLUKernelTest() override = default;

  // Test SiLU operation with different shapes
  bool SiLUFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -5, 5, kFloat16, kCPU);

      // 2. Compute reference result (FP16) on CPU
      // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          // Convert FP16 to FP32 for computation
          float x_val = MLLM_FP16_TO_FP32(x_ptr[i]);

          // Compute sigmoid(x) = 1 / (1 + exp(-x))
          float sigmoid_x;
          if (x_val >= 0) {
            sigmoid_x = 1.0f / (1.0f + std::exp(-x_val));
          } else {
            float exp_x = std::exp(x_val);
            sigmoid_x = exp_x / (1.0f + exp_x);
          }

          // SiLU(x) = x * sigmoid(x)
          float result = x_val * sigmoid_x;

          // Convert back to FP16
          r_ptr[i] = MLLM_FP32_TO_FP16(result);
        }
      }

      // 3. Move inputs to Ascend and run SiLU
      auto x_ascend = x_cpu.to(kAscend);
      auto y_ascend = mllm::nn::functional::silu(x_ascend);

      // 4. Move result back to CPU and compare with reference using allClose
      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        return false;
      }
    }
    return true;
  }
};