// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"

class PermuteKernelTest : public KernelTest {
 public:
  PermuteKernelTest() = default;
  ~PermuteKernelTest() override = default;

  bool test2DPermutation() {
    using namespace mllm;  // NOLINT

    // Test cases with different shapes and permutations
    std::vector<std::pair<std::vector<int>, std::vector<int>>> test_cases = {
        {{4, 4}, {1, 0}},             // Simple 2D transpose
        {{8, 16}, {1, 0}},            // Non-square 2D transpose
        {{32, 64}, {1, 0}},           // Larger 2D transpose
        {{2, 3, 4}, {2, 1, 0}},       // 3D reverse order
        {{2, 3, 4}, {0, 2, 1}},       // 3D partial permutation
        {{2, 3, 4, 5}, {0, 1, 3, 2}}  // 4D partial permutation
    };

    for (auto& test_case : test_cases) {
      auto shape = test_case.first;
      auto permute_axes = test_case.second;

      // Create random input tensor
      Tensor input = Tensor::random(shape, kFloat32, kCPU);

      // Compute reference result
      auto output_shape = shape;
      for (int i = 0; i < (int)shape.size(); ++i) { output_shape[i] = shape[permute_axes[i]]; }
      Tensor reference_output = Tensor::zeros(output_shape, kFloat32, kCPU);

      // Manual permutation implementation for reference
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto ref_ptr = reference_output.ptr<mllm_fp32_t>();

      // Calculate strides for input tensor
      std::vector<int> input_strides(shape.size());
      input_strides[shape.size() - 1] = 1;
      for (int i = (int)shape.size() - 2; i >= 0; --i) { input_strides[i] = input_strides[i + 1] * shape[i + 1]; }

      // Calculate strides for output tensor
      std::vector<int> output_strides(output_shape.size());
      output_strides[output_shape.size() - 1] = 1;
      for (int i = (int)output_shape.size() - 2; i >= 0; --i) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
      }

      // Perform permutation
      std::vector<int> indices(shape.size(), 0);
      for (size_t i = 0; i < input.numel(); ++i) {
        // Calculate input offset
        int input_offset = 0;
        for (size_t j = 0; j < indices.size(); ++j) { input_offset += indices[j] * input_strides[j]; }

        // Calculate output offset based on permutation
        int output_offset = 0;
        for (size_t j = 0; j < permute_axes.size(); ++j) { output_offset += indices[permute_axes[j]] * output_strides[j]; }

        ref_ptr[output_offset] = input_ptr[input_offset];

        // Update indices
        for (int j = (int)indices.size() - 1; j >= 0; --j) {
          indices[j]++;
          if (indices[j] < shape[j]) { break; }
          indices[j] = 0;
        }
      }

      // Compute using mllm permute
      Tensor output = input.permute(permute_axes);

      // Compare results
      auto result = test::allClose(output, reference_output);
      if (!result) {
        print(result);
        return false;
      }
    }

    return true;
  }
};
