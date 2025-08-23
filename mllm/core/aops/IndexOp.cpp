// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/IndexOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

IndexOp::IndexOp(const IndexOpOptions& options) : BaseOp(OpTypes::kIndex), options_(options) {}

void IndexOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void IndexOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::IndexOp>(shared_from_this(), i_irs, o_irs);
}

void IndexOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];
  auto shape = input.shape();
  auto& indices = options_.indices_;

  bool all_slices = true;
  for (auto& index : indices) { all_slices &= index.slice_indices_.has_value(); }

  if (all_slices) {
    NYI("Pls use slice op instead of index op");
    outputs.emplace_back(Tensor::nil());
    return;
  }

  MLLM_RT_ASSERT(input.isContiguous());
  MLLM_RT_ASSERT(output.isContiguous());

  auto output_shape = output.shape();
  size_t element_size = input.bytes() / input.numel();

  // Create index vectors for iteration
  std::vector<int> out_indices(output_shape.size(), 0);
  std::vector<int> in_indices(shape.size(), 0);

  size_t total_output_elements = 1;
  for (auto dim : output_shape) { total_output_elements *= dim; }

  // Precompute tensor index information
  std::vector<const int32_t*> tensor_ptrs;
  std::vector<Tensor::shape_t> tensor_shapes;

  for (const auto& index : indices) {
    if (index.tensor_indices_.has_value()) {
      const auto& tensor = index.tensor_indices_.value();
      tensor_ptrs.push_back(tensor.ptr<int32_t>());
      tensor_shapes.push_back(tensor.shape());
    } else {
      tensor_ptrs.push_back(nullptr);
      tensor_shapes.emplace_back();
    }
  }

  // Iterate through all output positions
  for (size_t i = 0; i < total_output_elements; ++i) {
    // Convert linear index to multi-dimensional indices
    size_t temp = i;
    for (int j = static_cast<int>(output_shape.size()) - 1; j >= 0; --j) {
      out_indices[j] = temp % output_shape[j];
      temp /= output_shape[j];
    }

    // Map output indices to input indices based on indexing rules
    bool valid_index = true;
    size_t out_dim = 0;

    for (size_t j = 0; j < indices.size() && j < shape.size(); ++j) {
      const auto& index = indices[j];

      if (index.slice_indices_.has_value()) {
        // Slice indexing
        auto slice = index.slice_indices_.value();
        int start = slice.start_ == kAll ? 0 : slice.start_;
        int step = slice.step_;
        in_indices[j] = start + out_indices[out_dim] * step;
        out_dim++;

        // Check bounds
        if (in_indices[j] < 0 || in_indices[j] >= static_cast<int>(shape[j])) {
          valid_index = false;
          break;
        }
      } else if (index.vector_indices_.has_value()) {
        // Vector indexing
        const auto& vec = index.vector_indices_.value();
        if (out_dim >= out_indices.size() || out_indices[out_dim] >= vec.size()) {
          valid_index = false;
          break;
        }
        in_indices[j] = vec[out_indices[out_dim]];
        out_dim++;

        // Check bounds
        if (in_indices[j] < 0 || in_indices[j] >= static_cast<int>(shape[j])) {
          valid_index = false;
          break;
        }
      } else if (index.tensor_indices_.has_value()) {
        // Tensor indexing
        const auto& index_tensor = index.tensor_indices_.value();
        const auto& tensor_shape = tensor_shapes[j];

        // Calculate linear index into the index tensor
        size_t tensor_index = 0;
        size_t stride = 1;

        for (int k = static_cast<int>(tensor_shape.size()) - 1; k >= 0; --k) {
          if (out_dim >= out_indices.size()) {
            valid_index = false;
            break;
          }
          tensor_index += out_indices[out_dim] * stride;
          stride *= tensor_shape[k];
          out_dim++;
        }

        if (!valid_index) break;

        // Convert linear index to multi-dimensional indices for the index tensor
        std::vector<int> tensor_indices(tensor_shape.size());
        size_t temp_index = tensor_index;
        for (int k = static_cast<int>(tensor_shape.size()) - 1; k >= 0; --k) {
          tensor_indices[k] = temp_index % tensor_shape[k];
          temp_index /= tensor_shape[k];
        }

        // Get the index value from the tensor using at<int32_t>
        in_indices[j] = index_tensor.constAt<int32_t>(tensor_indices);

        // Check bounds
        if (in_indices[j] < 0 || in_indices[j] >= static_cast<int>(shape[j])) {
          valid_index = false;
          break;
        }
      }
    }

    // Copy data if indices are valid
    if (valid_index) {
      // Calculate input offset
      size_t in_offset = 0;
      size_t multiplier = 1;
      for (int j = static_cast<int>(shape.size()) - 1; j >= 0; --j) {
        in_offset += in_indices[j] * multiplier;
        multiplier *= shape[j];
      }

      // Calculate output offset
      size_t out_offset = 0;
      multiplier = 1;
      for (int j = static_cast<int>(output_shape.size()) - 1; j >= 0; --j) {
        out_offset += out_indices[j] * multiplier;
        multiplier *= output_shape[j];
      }

      // Copy element
      std::memcpy(output.ptr<char>() + out_offset * element_size, input.ptr<char>() + in_offset * element_size, element_size);
    }
  }
}

void IndexOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto shape = input.shape();
  auto& indices = options_.indices_;

  bool all_slices = true;
  for (auto& index : indices) { all_slices &= index.slice_indices_.has_value(); }

  if (all_slices) {
    NYI("Pls use slice op instead of index op");
    outputs.emplace_back(Tensor::nil());
    return;
  }

  // Check if match shape
  {
    int cnt = 0;
    for (auto& index : indices) {
      if (index.slice_indices_.has_value()) {
        cnt++;
        continue;
      }
      if (index.tensor_indices_.has_value()) {
        cnt += index.tensor_indices_->shape().size();
        continue;
      }
      if (index.vector_indices_.has_value()) {
        cnt++;
        continue;
      }
    }
    MLLM_RT_ASSERT_EQ(cnt, shape.size());
  }

  Tensor::shape_t o_shape;
  calculateOutputShape(input, o_shape);
  outputs.emplace_back(Tensor::empty(o_shape, input.dtype(), input.device()));
}

void IndexOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

void IndexOp::calculateOutputShape(const Tensor& input, Tensor::shape_t& o_shape) const {
  auto shape = input.shape();
  auto& indices = options_.indices_;

  for (size_t i = 0; i < indices.size(); ++i) {
    auto& index = indices[i];
    if (index.slice_indices_.has_value()) {
      auto slice = index.slice_indices_.value();
      int start = slice.start_ == kAll ? 0 : slice.start_;
      int end = slice.end_ == kAll ? shape[i] : slice.end_;
      int step = slice.step_;
      // Calculate output dimension size
      int dim_size = (end - start + step - 1) / step;
      o_shape.push_back(dim_size);
    } else if (index.vector_indices_.has_value()) {
      // When indexed by a vector, the output dimension size equals vector size
      o_shape.push_back(index.vector_indices_.value().size());
    } else if (index.tensor_indices_.has_value()) {
      // When indexed by a tensor, the output dimension size equals tensor shape
      auto tensor_shape = index.tensor_indices_.value().shape();
      o_shape.insert(o_shape.end(), tensor_shape.begin(), tensor_shape.end());
    }
  }
}

void IndexOp::processTensorIndices(const Tensor& input, const std::vector<int>& out_indices, std::vector<int>& in_indices,
                                   bool& valid_index) const {
  // Track which output dimensions correspond to tensor indices
  int out_dim = 0;

  for (size_t j = 0; j < options_.indices_.size(); ++j) {
    const auto& index = options_.indices_[j];

    if (index.tensor_indices_.has_value()) {
      const auto& index_tensor = index.tensor_indices_.value();
      const auto& tensor_shape = index_tensor.shape();

      // For tensor indices, we need to extract values from the index tensor
      // based on the current output position
      std::vector<int> tensor_indices(tensor_shape.size());

      // Extract indices for the index tensor from the output indices
      for (size_t k = 0; k < tensor_shape.size(); ++k) {
        if (out_dim >= static_cast<int>(out_indices.size())) {
          valid_index = false;
          return;
        }
        tensor_indices[k] = out_indices[out_dim++];
      }

      // Get the actual index value from the index tensor
      int tensor_value = index_tensor.constAt<int32_t>(tensor_indices);
      in_indices[j] = tensor_value;

      // Check bounds
      if (in_indices[j] < 0 || in_indices[j] >= static_cast<int>(input.shape()[j])) {
        valid_index = false;
        return;
      }
    } else {
      // For non-tensor indices, just move to the next output dimension
      out_dim++;
    }
  }
}

}  // namespace mllm::aops
