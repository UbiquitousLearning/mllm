// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/nn/Functional.hpp"
#include "KernelTestHelper.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"
#include <vector>
#include <cmath>
#include <iostream>


class AscendLinearKernelTest : public KernelTest {
 public:
  AscendLinearKernelTest() = default;
  ~AscendLinearKernelTest() override = default;

  bool LinearFloat16Test(const std::vector<std::tuple<mllm::Tensor::shape_t, int, int>>& test_cases) {
    using namespace mllm;  // NOLINT
    for (auto& test_case : test_cases) {
      auto input_shape = std::get<0>(test_case);
      int in_channels = std::get<1>(test_case);
      int out_channels = std::get<2>(test_case);

      std::cout << "[LinearTest] Testing shape=[";
      for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i] << (i < input_shape.size() - 1 ? ", " : "");
      }
      std::cout << "], in=" << in_channels << ", out=" << out_channels << std::endl;

      // 1. Construct random FP16 inputs on CPU
      // x: [M, K] where K = in_channels
      Tensor x_cpu = Tensor::random(input_shape, -1, 1, kFloat16, kCPU);

      // Weight shape for ATB: [K, N] where K=in_channels, N=out_channels
      Tensor weight_cpu = Tensor::random({in_channels, out_channels}, -0.5, 0.5, kFloat16, kCPU);

      // 2. Compute reference result on CPU
      // y = x @ weight, where x is [M, K], weight is [K, N], output is [M, N]
      auto output_shape = input_shape;
      output_shape[output_shape.size() - 1] = out_channels;
      Tensor ref_cpu = Tensor::zeros(output_shape, kFloat16, kCPU);

      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* w_ptr = weight_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();

        size_t batch_size = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
          batch_size *= input_shape[i];
        }

        for (size_t b = 0; b < batch_size; ++b) {
          for (int o = 0; o < out_channels; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < in_channels; ++i) {
              float x_val = MLLM_FP16_TO_FP32(x_ptr[b * in_channels + i]);
              float w_val = MLLM_FP16_TO_FP32(w_ptr[i * out_channels + o]);  // weight is [K, N]
              sum += x_val * w_val;
            }
            r_ptr[b * out_channels + o] = MLLM_FP32_TO_FP16(sum);
          }
        }
      }

      // 3. Move inputs to Ascend and run Linear via matmul
      auto x_ascend = x_cpu.to(kAscend);
      auto weight_ascend = weight_cpu.to(kAscend);

      // Use matmul: y = x @ weight
      auto y_ascend = nn::functional::matmul(x_ascend, weight_ascend, false, false);

      // 4. Move result back to CPU and compare with reference
      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        std::cout << "[LinearTest] FAILED!" << std::endl;
        return false;
      }
      std::cout << "[LinearTest] PASSED" << std::endl;
    }
    return true;
  }


  bool LinearWithBiasFloat16Test(const std::vector<std::tuple<mllm::Tensor::shape_t, int, int>>& test_cases) {
    using namespace mllm;  // NOLINT
    for (auto& test_case : test_cases) {
      auto input_shape = std::get<0>(test_case);
      int in_channels = std::get<1>(test_case);
      int out_channels = std::get<2>(test_case);

      std::cout << "[LinearWithBiasTest] Testing shape=[";
      for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i] << (i < input_shape.size() - 1 ? ", " : "");
      }
      std::cout << "], in=" << in_channels << ", out=" << out_channels << std::endl;

      // 1. Create random input, weight and bias on CPU
      Tensor x_cpu = Tensor::random(input_shape, -1, 1, kFloat16, kCPU);
      // Weight shape: [out_channels, in_channels]
      Tensor weight_cpu = Tensor::random({out_channels, in_channels}, -0.5, 0.5, kFloat16, kCPU);
      // Bias shape: [1, out_channels] for ATB Linear (2D tensor required)
      Tensor bias_cpu = Tensor::random({1, out_channels}, -0.1, 0.1, kFloat16, kCPU);

      // 2. Compute reference result on CPU
      auto output_shape = input_shape;
      output_shape[output_shape.size() - 1] = out_channels;
      Tensor ref_cpu = Tensor::zeros(output_shape, kFloat16, kCPU);

      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* w_ptr = weight_cpu.ptr<mllm_fp16_t>();
        auto* b_ptr = bias_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();

        size_t batch_size = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
          batch_size *= input_shape[i];
        }

        // y = x @ W^T + b, where W is [out_channels, in_channels]
        for (size_t b = 0; b < batch_size; ++b) {
          for (int o = 0; o < out_channels; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < in_channels; ++i) {
              float x_val = MLLM_FP16_TO_FP32(x_ptr[b * in_channels + i]);
              float w_val = MLLM_FP16_TO_FP32(w_ptr[o * in_channels + i]);
              sum += x_val * w_val;
            }
            float bias_val = MLLM_FP16_TO_FP32(b_ptr[o]);
            sum += bias_val;
            r_ptr[b * out_channels + o] = MLLM_FP32_TO_FP16(sum);
          }
        }
      }

      // 3. Move tensors to Ascend and run linear
      auto x_ascend = x_cpu.to(kAscend);
      auto weight_ascend = weight_cpu.to(kAscend);
      auto bias_ascend = bias_cpu.to(kAscend);

      // Use nn::functional::linear directly
      auto y_ascend = nn::functional::linear(x_ascend, weight_ascend, bias_ascend);

      // 4. Compare result with reference
      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        std::cout << "[LinearWithBiasTest] FAILED!" << std::endl;
        return false;
      }
      std::cout << "[LinearWithBiasTest] PASSED" << std::endl;
    }
    return true;
  }
};
