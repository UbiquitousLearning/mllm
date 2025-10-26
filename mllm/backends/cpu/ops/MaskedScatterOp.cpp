// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/MaskedScatterOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/OpTypes.hpp"
#include <cstring>
#include <vector>
#include <numeric>

namespace mllm::cpu {

CPUMaskedScatterOp::CPUMaskedScatterOp(const aops::MaskedScatterOpOptions& options) : aops::MaskedScatterOp(options) {}

// Helper function to calculate broadcast shape
std::vector<int32_t> calculateBroadcastShape(const std::vector<int32_t>& shape_a, const std::vector<int32_t>& shape_b) {
  // Determine the maximum number of dimensions
  size_t max_ndim = std::max(shape_a.size(), shape_b.size());
  std::vector<int32_t> broadcast_shape(max_ndim);

  // Pad shapes to the same number of dimensions
  std::vector<int32_t> padded_a(max_ndim, 1);
  std::vector<int32_t> padded_b(max_ndim, 1);

  // Copy original shapes to the end (right-aligned)
  std::copy(shape_a.begin(), shape_a.end(), padded_a.begin() + (max_ndim - shape_a.size()));
  std::copy(shape_b.begin(), shape_b.end(), padded_b.begin() + (max_ndim - shape_b.size()));

  // Calculate broadcast shape
  for (size_t i = 0; i < max_ndim; ++i) {
    if (padded_a[i] == padded_b[i]) {
      broadcast_shape[i] = padded_a[i];
    } else if (padded_a[i] == 1) {
      broadcast_shape[i] = padded_b[i];
    } else if (padded_b[i] == 1) {
      broadcast_shape[i] = padded_a[i];
    } else {
      // Cannot broadcast, should not happen in valid cases
      broadcast_shape[i] = std::max(padded_a[i], padded_b[i]);
    }
  }

  return broadcast_shape;
}

// Helper function to calculate strides for broadcasting
std::vector<int32_t> calculateStrides(const std::vector<int32_t>& shape) {
  std::vector<int32_t> strides(shape.size(), 1);
  for (int i = shape.size() - 2; i >= 0; --i) { strides[i] = strides[i + 1] * shape[i + 1]; }
  return strides;
}

// Helper function to convert multi-dimensional index to linear index
int32_t getLinearIndex(const std::vector<int32_t>& indices, const std::vector<int32_t>& strides) {
  int32_t linear_index = 0;
  for (size_t i = 0; i < indices.size(); ++i) { linear_index += indices[i] * strides[i]; }
  return linear_index;
}

// Helper function to convert linear index to multi-dimensional indices
std::vector<int32_t> getMultiDimIndices(int32_t linear_index, const std::vector<int32_t>& shape) {
  std::vector<int32_t> indices(shape.size());
  for (int i = shape.size() - 1; i >= 0; --i) {
    indices[i] = linear_index % shape[i];
    linear_index /= shape[i];
  }
  return indices;
}

void CPUMaskedScatterOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // dst, mask, src
  auto& dst = inputs[0];
  auto& mask = inputs[1];
  auto& src = inputs[2];

  MLLM_RT_ASSERT_EQ(mask.dtype(), kInt8);

  // dst and output should be the same tensor (in-place operation)
  // But we still need to ensure output has the correct shape
  auto& output = outputs[0];

  // Get shapes
  auto dst_shape = dst.shape();
  auto mask_shape = mask.shape();
  auto src_shape = src.shape();

  // Calculate broadcast shape for all tensors
  auto broadcast_shape = calculateBroadcastShape(dst_shape, mask_shape);
  broadcast_shape = calculateBroadcastShape(broadcast_shape, src_shape);

  // Calculate strides for broadcasting
  auto dst_strides = calculateStrides(dst_shape);
  auto mask_strides = calculateStrides(mask_shape);
  auto src_strides = calculateStrides(src_shape);
  auto broadcast_strides = calculateStrides(broadcast_shape);

  // Calculate total elements
  int32_t total_elements = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<>());

  // Check data types
  MLLM_RT_ASSERT(dst.dtype() == src.dtype());

  // Handle different data types
  if (dst.dtype() == MLLM_TYPE_F32) {
    float* dst_ptr = dst.ptr<float>();
    float* src_ptr = src.ptr<float>();
    uint8_t* mask_ptr = mask.ptr<uint8_t>();

    for (int32_t i = 0; i < total_elements; ++i) {
      // Convert linear index to multi-dimensional indices
      auto indices = getMultiDimIndices(i, broadcast_shape);

      // Calculate index for mask with broadcasting
      int32_t mask_linear_index = 0;
      if (mask_shape.size() == 1 && mask_shape[0] == 1) {
        // Scalar mask
        mask_linear_index = 0;
      } else {
        // Multi-dimensional mask
        std::vector<int32_t> mask_indices(mask_shape.size());
        int offset = broadcast_shape.size() - mask_shape.size();
        for (size_t j = 0; j < mask_shape.size(); ++j) { mask_indices[j] = indices[j + offset] % mask_shape[j]; }
        mask_linear_index = getLinearIndex(mask_indices, mask_strides);
      }

      // If mask is true, copy from src to dst
      if (mask_ptr[mask_linear_index] != 0) {
        // Calculate index for dst
        int32_t dst_linear_index = 0;
        if (dst_shape.size() == 1 && dst_shape[0] == 1) {
          // Scalar dst
          dst_linear_index = 0;
        } else {
          // Multi-dimensional dst
          std::vector<int32_t> dst_indices(dst_shape.size());
          int offset = broadcast_shape.size() - dst_shape.size();
          for (size_t j = 0; j < dst_shape.size(); ++j) { dst_indices[j] = indices[j + offset] % dst_shape[j]; }
          dst_linear_index = getLinearIndex(dst_indices, dst_strides);
        }

        // Calculate index for src with broadcasting
        int32_t src_linear_index = 0;
        if (src_shape.size() == 1 && src_shape[0] == 1) {
          // Scalar src
          src_linear_index = 0;
        } else {
          // Multi-dimensional src
          std::vector<int32_t> src_indices(src_shape.size());
          int offset = broadcast_shape.size() - src_shape.size();
          for (size_t j = 0; j < src_shape.size(); ++j) { src_indices[j] = indices[j + offset] % src_shape[j]; }
          src_linear_index = getLinearIndex(src_indices, src_strides);
        }

        dst_ptr[dst_linear_index] = src_ptr[src_linear_index];
      }
    }
  } else if (dst.dtype() == MLLM_TYPE_F16) {
    mllm_fp16_t* dst_ptr = dst.ptr<mllm_fp16_t>();
    mllm_fp16_t* src_ptr = src.ptr<mllm_fp16_t>();
    uint8_t* mask_ptr = mask.ptr<uint8_t>();

    for (int32_t i = 0; i < total_elements; ++i) {
      // Convert linear index to multi-dimensional indices
      auto indices = getMultiDimIndices(i, broadcast_shape);

      // Calculate index for mask with broadcasting
      int32_t mask_linear_index = 0;
      if (mask_shape.size() == 1 && mask_shape[0] == 1) {
        // Scalar mask
        mask_linear_index = 0;
      } else {
        // Multi-dimensional mask
        std::vector<int32_t> mask_indices(mask_shape.size());
        int offset = broadcast_shape.size() - mask_shape.size();
        for (size_t j = 0; j < mask_shape.size(); ++j) { mask_indices[j] = indices[j + offset] % mask_shape[j]; }
        mask_linear_index = getLinearIndex(mask_indices, mask_strides);
      }

      // If mask is true, copy from src to dst
      if (mask_ptr[mask_linear_index] != 0) {
        // Calculate index for dst
        int32_t dst_linear_index = 0;
        if (dst_shape.size() == 1 && dst_shape[0] == 1) {
          // Scalar dst
          dst_linear_index = 0;
        } else {
          // Multi-dimensional dst
          std::vector<int32_t> dst_indices(dst_shape.size());
          int offset = broadcast_shape.size() - dst_shape.size();
          for (size_t j = 0; j < dst_shape.size(); ++j) { dst_indices[j] = indices[j + offset] % dst_shape[j]; }
          dst_linear_index = getLinearIndex(dst_indices, dst_strides);
        }

        // Calculate index for src with broadcasting
        int32_t src_linear_index = 0;
        if (src_shape.size() == 1 && src_shape[0] == 1) {
          // Scalar src
          src_linear_index = 0;
        } else {
          // Multi-dimensional src
          std::vector<int32_t> src_indices(src_shape.size());
          int offset = broadcast_shape.size() - src_shape.size();
          for (size_t j = 0; j < src_shape.size(); ++j) { src_indices[j] = indices[j + offset] % src_shape[j]; }
          src_linear_index = getLinearIndex(src_indices, src_strides);
        }

        dst_ptr[dst_linear_index] = src_ptr[src_linear_index];
      }
    }
  } else {
    // For other data types, we could add support similarly
    NYI("Unsupported data type for MaskedScatter operation");
  }

  // Copy result to output
  if (output.ptr<void>() != dst.ptr<void>()) { std::memcpy(output.ptr<void>(), dst.ptr<void>(), dst.bytes()); }
}

}  // namespace mllm::cpu
