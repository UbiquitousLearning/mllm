// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"

class TopKKernelTest : public KernelTest {
 public:
  TopKKernelTest() = default;
  ~TopKKernelTest() override = default;

  bool testTopK(const std::vector<mllm::Tensor::shape_t>& shapes, int k, int dim = -1) {
    using namespace mllm;  // NOLINT

    for (const auto& shape : shapes) {
      // Create input tensor with random values
      Tensor input = Tensor::random(shape, -10.0, 10.0, kFloat32, kCPU);

      // Apply TopK operation
      auto outputs = nn::functional::topk(input, k, dim);
      const Tensor& values = outputs[0];
      const Tensor& indices = outputs[1];

      // Verify the output shapes
      auto expected_shape = shape;
      if (dim < 0) {
        expected_shape[expected_shape.size() + dim] = k;
      } else {
        expected_shape[dim] = k;
      }

      if (values.shape() != expected_shape) { return false; }

      if (indices.shape() != expected_shape) { return false; }

      // Verify data types
      if (values.dtype() != kFloat32) { return false; }

      if (indices.dtype() != kInt32) { return false; }

      // Verify that values are actually the top k values (in descending order)
      auto values_ptr = values.ptr<float>();
      auto indices_ptr = indices.ptr<int32_t>();

      // Calculate sizes for validation
      int test_dim = dim;
      if (test_dim < 0) { test_dim += shape.size(); }

      int outer_size = 1;
      int inner_size = 1;
      for (int i = 0; i < test_dim; ++i) { outer_size *= shape[i]; }
      for (int i = test_dim + 1; i < shape.size(); ++i) { inner_size *= shape[i]; }

      // Check that values are in descending order for each slice
      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
          for (int i = 1; i < k; ++i) {
            int index = out * k * inner_size + i * inner_size + in;
            int prev_index = out * k * inner_size + (i - 1) * inner_size + in;
            // Check that values are in descending order (for largest=true, which is default)
            if (values_ptr[index] > values_ptr[prev_index]) { return false; }
          }
        }
      }
    }
    return true;
  }
};
