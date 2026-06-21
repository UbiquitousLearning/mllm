// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"

class TransposeKernelTest : public KernelTest {
 public:
  TransposeKernelTest() = default;
  ~TransposeKernelTest() override = default;

  bool testHWTransposition() {
    using namespace mllm;  // NOLINT

    // Test cases with different shapes
    std::vector<std::pair<int, int>> test_shapes = {{4, 4}, {8, 16}, {16, 8}, {32, 64}, {128, 256}};

    for (auto& shape : test_shapes) {
      int H = shape.first;
      int W = shape.second;

      // Create random input tensor
      Tensor input = Tensor::random({H, W}, -1, 1, kFloat32, kCPU);

      // Compute reference result
      Tensor reference_output = Tensor::zeros({W, H}, kFloat32, kCPU);
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto ref_ptr = reference_output.ptr<mllm_fp32_t>();

      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) { ref_ptr[w * H + h] = input_ptr[h * W + w]; }
      }

      // Compute using mllm transpose
      Tensor output = input.transpose(0, 1);

      // Compare results
      auto result = test::allClose(output, reference_output);
      if (!result) {
        print(result);
        return false;
      }
    }

    return true;
  }

  bool testBSHDTransposition() {
    using namespace mllm;  // NOLINT

    // Test cases with different shapes
    std::vector<std::array<int, 4>> test_shapes = {{2, 4, 4, 8}, {1, 8, 16, 32}, {3, 16, 8, 16}, {1, 32, 64, 128}};

    for (auto& shape : test_shapes) {
      int B = shape[0];
      int S = shape[1];
      int H = shape[2];
      int D = shape[3];

      // Create random input tensor
      Tensor input = Tensor::random({B, S, H, D}, -1, 1, kFloat32, kCPU);

      // Compute reference result (transpose S and H dimensions)
      Tensor reference_output = Tensor::zeros({B, H, S, D}, kFloat32, kCPU);
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto ref_ptr = reference_output.ptr<mllm_fp32_t>();

      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
          for (int h = 0; h < H; ++h) {
            for (int d = 0; d < D; ++d) { ref_ptr[((b * H + h) * S + s) * D + d] = input_ptr[((b * S + s) * H + h) * D + d]; }
          }
        }
      }

      // Compute using mllm transpose (dimensions 1 and 2)
      Tensor output = input.transpose(1, 2);

      // Compare results
      auto result = test::allClose(output, reference_output);
      if (!result) {
        print(result);
        return false;
      }
    }

    return true;
  }

  bool testGeneralTransposition() {
    using namespace mllm;  // NOLINT

    // Test general transposition with 3D tensor
    // {4, 8, 16}, {8, 16, 32}, {16, 32, 64},
    std::vector<std::array<int, 3>> test_shapes = {{1, 464, 513}};

    for (auto& shape : test_shapes) {
      int D0 = shape[0];
      int D1 = shape[1];
      int D2 = shape[2];

      // Create random input tensor
      Tensor input = Tensor::random({D0, D1, D2}, -1, 1, kFloat32, kCPU);

      // Test transposing last two dimensions (-1 and -2)
      // Compute reference result
      Tensor reference_output = Tensor::zeros({D0, D2, D1}, kFloat32, kCPU);
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto ref_ptr = reference_output.ptr<mllm_fp32_t>();

      for (int i = 0; i < D0; ++i) {
        for (int j = 0; j < D1; ++j) {
          for (int k = 0; k < D2; ++k) { ref_ptr[i * D2 * D1 + k * D1 + j] = input_ptr[i * D1 * D2 + j * D2 + k]; }
        }
      }
      // Compute using mllm transpose (dimensions -1 and -2)
      Tensor output = input.T();

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
