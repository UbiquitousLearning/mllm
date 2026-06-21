// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"

class ClipKernelTest : public KernelTest {
 public:
  ClipKernelTest() = default;
  ~ClipKernelTest() override = default;

  bool testClip(const std::vector<mllm::Tensor::shape_t>& shapes, float min_val, float max_val) {
    using namespace mllm;  // NOLINT

    for (const auto& shape : shapes) {
      // Create input tensor with random values that span a wide range
      Tensor input = Tensor::random(shape, -20.0, 20.0, kFloat32, kCPU);

      // Apply Clip operation
      Tensor output = nn::functional::clip(input, min_val, max_val);

      // Verify the output shape
      if (output.shape() != shape) { return false; }

      // Verify data type
      if (output.dtype() != kFloat32) { return false; }

      // Verify that all values are within the clip range
      auto input_ptr = input.ptr<float>();
      auto output_ptr = output.ptr<float>();
      auto numel = output.numel();

      for (size_t i = 0; i < numel; ++i) {
        if (output_ptr[i] < min_val || output_ptr[i] > max_val) { return false; }

        // Check that clipping was performed correctly
        if (input_ptr[i] < min_val && output_ptr[i] != min_val) { return false; }

        if (input_ptr[i] > max_val && output_ptr[i] != max_val) { return false; }

        if (input_ptr[i] >= min_val && input_ptr[i] <= max_val && output_ptr[i] != input_ptr[i]) { return false; }
      }
    }
    return true;
  }
};
