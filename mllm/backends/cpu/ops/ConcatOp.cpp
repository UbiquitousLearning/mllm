// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>

#include "mllm/backends/cpu/ops/ConcatOp.hpp"

namespace mllm::cpu {

CPUConcatOp::CPUConcatOp(const aops::ConcatOpOptions& options) : aops::ConcatOp(options) {}

void CPUConcatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  bool is_all_contiguous = true;
  for (auto& input : inputs) { is_all_contiguous &= input.isContiguous(); }

  auto concat_at_dim = options_.dim;

  if (concat_at_dim < 0) { concat_at_dim += inputs[0].rank(); }

  // If not contiguous, we can not copy things using simd, but one by one. we need to give an
  // warning to user, because it is slow.

  if (is_all_contiguous) {
    // Calculate base size (elements after concat dim)
    int base_size = 1;
    for (int i = concat_at_dim + 1; i < inputs[0].shape().size(); ++i) { base_size *= inputs[0].shape()[i]; }

    // Calculate number of slices (elements before concat dim)
    int num_slices = 1;
    for (int i = 0; i < concat_at_dim; ++i) { num_slices *= inputs[0].shape()[i]; }

    // Calculate total concat dim size
    int total_concat_dim_size = 0;
    for (auto& input : inputs) { total_concat_dim_size += input.shape()[concat_at_dim]; }

    switch (outputs[0].dtype()) {
      case kFloat32: {
        mllm_fp32_t* out_ptr = outputs[0].ptr<mllm_fp32_t>();
        int current_offset = 0;
        for (auto& input : inputs) {
          int current_concat_size = input.shape()[concat_at_dim];
          int chunk_size = current_concat_size * base_size;
          const mllm_fp32_t* in_ptr = input.ptr<mllm_fp32_t>();
          for (int slice = 0; slice < num_slices; ++slice) {
            const mllm_fp32_t* src = in_ptr + slice * chunk_size;
            mllm_fp32_t* dst = out_ptr + slice * (total_concat_dim_size * base_size) + current_offset * base_size;
            std::memcpy(dst, src, chunk_size * sizeof(mllm_fp32_t));
          }
          current_offset += current_concat_size;
        }
        break;
      }
      case kFloat16: {
        mllm_fp16_t* out_ptr = outputs[0].ptr<mllm_fp16_t>();
        int current_offset = 0;
        for (auto& input : inputs) {
          int current_concat_size = input.shape()[concat_at_dim];
          int chunk_size = current_concat_size * base_size;
          const mllm_fp16_t* in_ptr = input.ptr<mllm_fp16_t>();

          for (int slice = 0; slice < num_slices; ++slice) {
            const mllm_fp16_t* src = in_ptr + slice * chunk_size;
            mllm_fp16_t* dst = out_ptr + slice * (total_concat_dim_size * base_size) + current_offset * base_size;
            std::memcpy(dst, src, chunk_size * sizeof(mllm_fp16_t));
          }
          current_offset += current_concat_size;
        }
        break;
      }
      default: NYI("Type not supported in concat op");
    }
  } else {
    MLLM_WARN("Concat op has weak performance for non-contiguous inputs.");

    std::vector<int> input_concat_offsets;
    int current_offset = 0;
    for (auto& input : inputs) {
      input_concat_offsets.push_back(current_offset);
      current_offset += input.shape()[concat_at_dim];
    }

    switch (outputs[0].dtype()) {
      case kFloat32: {
        for (int i = 0; i < inputs.size(); ++i) {
          auto input = inputs[i];
          int input_concat_offset = input_concat_offsets[i];
          std::vector<int> input_shape = input.shape();

          for (int64_t j = 0; j < input.numel(); ++j) {
            std::vector<int> input_index(input_shape.size(), 0);
            int64_t temp = j;
            for (int k = input_shape.size() - 1; k >= 0; --k) {
              input_index[k] = temp % input_shape[k];
              temp /= input_shape[k];
            }

            std::vector<int> output_index = input_index;
            output_index[concat_at_dim] = input_concat_offset + input_index[concat_at_dim];

            mllm_fp32_t value = input.at<mllm_fp32_t>(input_index);
            outputs[0].at<mllm_fp32_t>(output_index) = value;
          }
        }
        break;
      }
      case kFloat16: {
        for (int i = 0; i < inputs.size(); ++i) {
          auto input = inputs[i];
          int input_concat_offset = input_concat_offsets[i];
          std::vector<int> input_shape = input.shape();

          for (int64_t j = 0; j < input.numel(); ++j) {
            std::vector<int> input_index(input_shape.size(), 0);
            int64_t temp = j;
            for (int k = input_shape.size() - 1; k >= 0; --k) {
              input_index[k] = temp % input_shape[k];
              temp /= input_shape[k];
            }

            std::vector<int> output_index = input_index;
            output_index[concat_at_dim] = input_concat_offset + input_index[concat_at_dim];

            mllm_fp16_t value = input.at<mllm_fp16_t>(input_index);
            outputs[0].at<mllm_fp16_t>(output_index) = value;
          }
        }
        break;
      }
      default: NYI("Type not supported in concat op");
    }
  }
}

}  // namespace mllm::cpu
