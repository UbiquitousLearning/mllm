// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/layers/RMSNorm.hpp"
#include "KernelTestHelper.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"
#include <vector>
#include <cmath>

class AscendRMSNormKernelTest : public KernelTest {
 public:
  AscendRMSNormKernelTest() = default;
  ~AscendRMSNormKernelTest() override = default;

  // Test RMSNorm operation with different shapes
  bool RMSNormFloat16Test(const std::vector<std::tuple<mllm::Tensor::shape_t, int, float>>& test_cases) {
    using namespace mllm;  // NOLINT
    for (auto& test_case : test_cases) {
      auto input_shape = std::get<0>(test_case);
      int norm_size = std::get<1>(test_case);
      float epsilon = std::get<2>(test_case);

      // Validate that norm_size matches the last dimension of input_shape
      assert(norm_size == static_cast<int>(input_shape.back()) &&
             "norm_size must equal the last dimension of input_shape");

      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(input_shape, -2, 2, kFloat16, kCPU);

      // Weight shape: [norm_size]
      Tensor weight_cpu = Tensor::random({norm_size}, 0.5, 1.5, kFloat16, kCPU);

      // 2. Compute reference result (FP16) on CPU
      // RMSNorm: y = x * weight / sqrt(mean(x^2) + epsilon)
      Tensor ref_cpu = Tensor::zeros(input_shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* w_ptr = weight_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();

        size_t batch_size = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
          batch_size *= input_shape[i];
        }

        // Perform RMSNorm for each batch
        for (size_t b = 0; b < batch_size; ++b) {
          float sum_squares = 0.0f;
          for (int i = 0; i < norm_size; ++i) {
            float x_val = MLLM_FP16_TO_FP32(x_ptr[b * norm_size + i]);
            sum_squares += x_val * x_val;
          }
          float rms = std::sqrt(sum_squares / norm_size + epsilon);

          // Normalize and scale by weight
          for (int i = 0; i < norm_size; ++i) {
            float x_val = MLLM_FP16_TO_FP32(x_ptr[b * norm_size + i]);
            float w_val = MLLM_FP16_TO_FP32(w_ptr[i]);
            float result = (x_val / rms) * w_val;
            r_ptr[b * norm_size + i] = MLLM_FP32_TO_FP16(result);
          }
        }
      }

      // 3. Move inputs to Ascend and run RMSNorm
      auto x_ascend = x_cpu.to(kAscend);
      auto weight_ascend = weight_cpu.to(kAscend);

      // Use functional API - one line to execute the operator
      auto y_ascend = nn::functional::rms_norm(x_ascend, weight_ascend, epsilon);

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
