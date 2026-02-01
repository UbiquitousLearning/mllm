// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "KernelTestHelper.hpp"

#include <vector>

class AscendKernelTest : public KernelTest {
 public:
  AscendKernelTest() = default;
  ~AscendKernelTest() override = default;

  // Test Add operation with different shapes
  bool AddFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);
      Tensor y_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);

      // 2. Compute reference result (FP16) on CPU
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* y_ptr = y_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          r_ptr[i] = x_ptr[i] + y_ptr[i];
        }
      }

      // 3. Move inputs to Ascend and run Add (z = x + y)
      auto x_ascend = x_cpu.to(kAscend);
      auto y_ascend = y_cpu.to(kAscend);
      auto z_ascend = x_ascend + y_ascend;

      // 4. Move result back to CPU and compare with reference using allClose
      auto z_cpu = z_ascend.to(kCPU);
      auto result = mllm::test::allClose(z_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        return false;
      }
    }
    return true;
  }

  // Test Sub operation with different shapes
  bool SubFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);
      Tensor y_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);

      // 2. Compute reference result (FP16) on CPU
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* y_ptr = y_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          r_ptr[i] = x_ptr[i] - y_ptr[i];
        }
      }

      // 3. Move inputs to Ascend and run Sub (z = x - y)
      auto x_ascend = x_cpu.to(kAscend);
      auto y_ascend = y_cpu.to(kAscend);
      auto z_ascend = x_ascend - y_ascend;

      // 4. Move result back to CPU and compare with reference using allClose
      auto z_cpu = z_ascend.to(kCPU);
      auto result = mllm::test::allClose(z_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        return false;
      }
    }
    return true;
  }

  // Test Mul operation with different shapes
  bool MulFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);
      Tensor y_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);

      // 2. Compute reference result (FP16) on CPU
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* y_ptr = y_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          r_ptr[i] = x_ptr[i] * y_ptr[i];
        }
      }

      // 3. Move inputs to Ascend and run Mul (z = x * y)
      auto x_ascend = x_cpu.to(kAscend);
      auto y_ascend = y_cpu.to(kAscend);
      auto z_ascend = x_ascend * y_ascend;

      // 4. Move result back to CPU and compare with reference using allClose
      auto z_cpu = z_ascend.to(kCPU);
      auto result = mllm::test::allClose(z_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        return false;
      }
    }
    return true;
  }
};

