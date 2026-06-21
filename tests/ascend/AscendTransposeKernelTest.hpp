// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "KernelTestHelper.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"
#include <vector>
#include <cmath>
#include <iostream>

class AscendTransposeKernelTest : public KernelTest {
 public:
  AscendTransposeKernelTest() = default;
  ~AscendTransposeKernelTest() override = default;

  // Test Transpose operation with different shapes and dimension pairs
  bool TransposeFloat16Test(const std::vector<std::tuple<mllm::Tensor::shape_t, int, int>>& test_cases) {
    using namespace mllm;  // NOLINT
    for (auto& test_case : test_cases) {
      auto shape = std::get<0>(test_case);
      int dim0 = std::get<1>(test_case);
      int dim1 = std::get<2>(test_case);

      std::cout << "[TransposeTest] Testing shape=[";
      for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
      }
      std::cout << "], dim0=" << dim0 << ", dim1=" << dim1 << std::endl;

      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -1, 1, kFloat16, kCPU);

      // 2. Compute reference result on CPU
      int ndim = static_cast<int>(shape.size());
      int d0 = dim0 < 0 ? dim0 + ndim : dim0;
      int d1 = dim1 < 0 ? dim1 + ndim : dim1;

      // Compute output shape
      auto out_shape = shape;
      std::swap(out_shape[d0], out_shape[d1]);

      Tensor ref_cpu = Tensor::zeros(out_shape, kFloat16, kCPU);

      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();

        // Compute strides for input tensor
        std::vector<size_t> in_strides(ndim);
        in_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
          in_strides[i] = in_strides[i + 1] * shape[i + 1];
        }

        // Compute strides for output tensor
        std::vector<size_t> out_strides(ndim);
        out_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
          out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        // Create permutation
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) {
          perm[i] = i;
        }
        std::swap(perm[d0], perm[d1]);

        // Iterate over all elements in output
        size_t total_elements = ref_cpu.numel();
        for (size_t out_idx = 0; out_idx < total_elements; ++out_idx) {
          // Convert linear index to multi-dimensional index in output
          std::vector<size_t> out_coords(ndim);
          size_t remaining = out_idx;
          for (int i = 0; i < ndim; ++i) {
            out_coords[i] = remaining / out_strides[i];
            remaining = remaining % out_strides[i];
          }

          // Map output coordinates to input coordinates using inverse permutation
          std::vector<size_t> in_coords(ndim);
          for (int i = 0; i < ndim; ++i) {
            in_coords[perm[i]] = out_coords[i];
          }

          // Convert input coordinates to linear index
          size_t in_idx = 0;
          for (int i = 0; i < ndim; ++i) {
            in_idx += in_coords[i] * in_strides[i];
          }

          r_ptr[out_idx] = x_ptr[in_idx];
        }
      }

      // 3. Move inputs to Ascend and run Transpose
      auto x_ascend = x_cpu.to(kAscend);
      auto y_ascend = x_ascend.transpose(dim0, dim1);

      // 4. Move result back to CPU and compare with reference
      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-5f, 1e-5f);
      if (!result.is_close) {
        std::cout << "[TransposeTest] FAILED!" << std::endl;
        return false;
      }
      std::cout << "[TransposeTest] PASSED" << std::endl;
    }
    return true;
  }
};
