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

class AscendSoftmaxKernelTest : public KernelTest {
 public:
  AscendSoftmaxKernelTest() = default;
  ~AscendSoftmaxKernelTest() override = default;

  // Test Softmax operation with different shapes and axes
  bool SoftmaxFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes, const std::vector<int>& axes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      for (auto axis : axes) {
        // 1. Construct random FP16 inputs on CPU
        Tensor x_cpu = Tensor::random(shape, -5, 5, kFloat16, kCPU);

        // 2. Compute reference result (FP16) on CPU
        // Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
        {
          auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
          auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();

          // Convert axis to positive index
          int ndim = static_cast<int>(shape.size());
          int pos_axis = axis;
          if (pos_axis < 0) {
            pos_axis = ndim + pos_axis;
          }

          size_t outer_size = 1;
          for (int i = 0; i < pos_axis; ++i) {
            outer_size *= shape[i];
          }

          size_t axis_size = shape[pos_axis];

          size_t inner_size = 1;
          for (int i = pos_axis + 1; i < ndim; ++i) {
            inner_size *= shape[i];
          }

          // Compute softmax for each slice along the axis
          for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
              auto idx_at = [&](size_t i) -> size_t {
                return (outer * axis_size + i) * inner_size + inner;
              };
              // Find max value for numerical stability
              float max_val = -std::numeric_limits<float>::infinity();
              for (size_t i = 0; i < axis_size; ++i) {
                size_t idx = idx_at(i);
                float val = MLLM_FP16_TO_FP32(x_ptr[idx]);
                max_val = std::max(max_val, val);
              }

              // Compute exp(x - max) and sum
              float sum_exp = 0.0f;
              std::vector<float> exp_vals(axis_size);
              for (size_t i = 0; i < axis_size; ++i) {
                size_t idx = idx_at(i);
                float val = MLLM_FP16_TO_FP32(x_ptr[idx]);
                exp_vals[i] = std::exp(val - max_val);
                sum_exp += exp_vals[i];
              }

              // Compute softmax and store result
              for (size_t i = 0; i < axis_size; ++i) {
                size_t idx = idx_at(i);
                float result = exp_vals[i] / sum_exp;
                r_ptr[idx] = MLLM_FP32_TO_FP16(result);
              }
            }
          }
        }

        // 3. Move inputs to Ascend and run Softmax
        auto x_ascend = x_cpu.to(kAscend);
        auto y_ascend = mllm::nn::functional::softmax(x_ascend, axis);

        // 4. Move result back to CPU and compare with reference using allClose
        auto y_cpu = y_ascend.to(kCPU);
        auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
        if (!result.is_close) {
          return false;
        }
      }
    }
    return true;
  }
};