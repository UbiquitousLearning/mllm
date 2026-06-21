// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>

#include "mllm/backends/cpu/ops/StackOp.hpp"

namespace mllm::cpu {

CPUStackOp::CPUStackOp(const aops::StackOpOptions& options) : aops::StackOp(options) {}

void CPUStackOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  bool is_all_contiguous = true;
  for (auto& input : inputs) { is_all_contiguous &= input.isContiguous(); }

  int stack_dim = options_.dim;
  const int input_rank = inputs[0].rank();
  if (stack_dim < 0) { stack_dim += (input_rank + 1); }

  const int N = static_cast<int>(inputs.size());

  if (is_all_contiguous) {
    // Elements before stack_dim
    int num_slices = 1;
    for (int i = 0; i < stack_dim; ++i) { num_slices *= inputs[0].shape()[i]; }

    // Elements after stack_dim in input (inner block size)
    int inner_size = 1;
    for (int i = stack_dim; i < input_rank; ++i) { inner_size *= inputs[0].shape()[i]; }

    switch (outputs[0].dtype()) {
      case kFloat32: {
        mllm_fp32_t* out_ptr = outputs[0].ptr<mllm_fp32_t>();
        for (int k = 0; k < N; ++k) {
          const mllm_fp32_t* in_ptr = inputs[k].ptr<mllm_fp32_t>();
          for (int slice = 0; slice < num_slices; ++slice) {
            const mllm_fp32_t* src = in_ptr + slice * inner_size;
            mllm_fp32_t* dst = out_ptr + (slice * N + k) * inner_size;
            std::memcpy(dst, src, inner_size * sizeof(mllm_fp32_t));
          }
        }
        break;
      }
      case kFloat16: {
        mllm_fp16_t* out_ptr = outputs[0].ptr<mllm_fp16_t>();
        for (int k = 0; k < N; ++k) {
          const mllm_fp16_t* in_ptr = inputs[k].ptr<mllm_fp16_t>();
          for (int slice = 0; slice < num_slices; ++slice) {
            const mllm_fp16_t* src = in_ptr + slice * inner_size;
            mllm_fp16_t* dst = out_ptr + (slice * N + k) * inner_size;
            std::memcpy(dst, src, inner_size * sizeof(mllm_fp16_t));
          }
        }
        break;
      }
      default: NYI("Type not supported in stack op");
    }
  } else {
    MLLM_WARN("Stack op has weak performance for non-contiguous inputs.");

    switch (outputs[0].dtype()) {
      case kFloat32: {
        for (int k = 0; k < N; ++k) {
          auto input = inputs[k];
          std::vector<int> input_shape = input.shape();

          for (int64_t j = 0; j < input.numel(); ++j) {
            std::vector<int> input_index(input_shape.size(), 0);
            int64_t temp = j;
            for (int d = input_shape.size() - 1; d >= 0; --d) {
              input_index[d] = temp % input_shape[d];
              temp /= input_shape[d];
            }

            std::vector<int> output_index;
            output_index.reserve(input_shape.size() + 1);
            for (int d = 0; d < stack_dim; ++d) { output_index.push_back(input_index[d]); }
            output_index.push_back(k);
            for (int d = stack_dim; d < input_shape.size(); ++d) { output_index.push_back(input_index[d]); }

            mllm_fp32_t value = input.at<mllm_fp32_t>(input_index);
            outputs[0].at<mllm_fp32_t>(output_index) = value;
          }
        }
        break;
      }
      case kFloat16: {
        for (int k = 0; k < N; ++k) {
          auto input = inputs[k];
          std::vector<int> input_shape = input.shape();

          for (int64_t j = 0; j < input.numel(); ++j) {
            std::vector<int> input_index(input_shape.size(), 0);
            int64_t temp = j;
            for (int d = input_shape.size() - 1; d >= 0; --d) {
              input_index[d] = temp % input_shape[d];
              temp /= input_shape[d];
            }

            std::vector<int> output_index;
            output_index.reserve(input_shape.size() + 1);
            for (int d = 0; d < stack_dim; ++d) { output_index.push_back(input_index[d]); }
            output_index.push_back(k);
            for (int d = stack_dim; d < input_shape.size(); ++d) { output_index.push_back(input_index[d]); }

            mllm_fp16_t value = input.at<mllm_fp16_t>(input_index);
            outputs[0].at<mllm_fp16_t>(output_index) = value;
          }
        }
        break;
      }
      default: NYI("Type not supported in stack op");
    }
  }
}

}  // namespace mllm::cpu