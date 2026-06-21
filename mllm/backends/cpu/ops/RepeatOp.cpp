// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/RepeatOp.hpp"

namespace mllm::cpu {

CPURepeatOp::CPURepeatOp(const aops::RepeatOpOptions& options) : aops::RepeatOp(options) {}

void CPURepeatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  auto multiplier = options_.repeat_times;
  auto dim = options_.dim;

  if (dim < 0 || dim >= static_cast<int>(X.shape().size())) {
    throw std::invalid_argument("CPURepeatOp::forward - invalid repeat dimension");
  }

  size_t outer_num = 1;
  for (int i = 0; i < dim; ++i) { outer_num *= X.shape()[i]; }

  size_t dim_size = X.shape()[dim];
  size_t inner_num = 1;
  for (int i = dim + 1; i < X.shape().size(); ++i) { inner_num *= X.shape()[i]; }

  size_t copy_size = inner_num * multiplier;
  size_t x_step = dim_size * inner_num;
  size_t y_step = dim_size * multiplier * inner_num;

  switch (X.dtype()) {
    case kFloat32: {
      const float* x_data = X.ptr<float>();
      float* y_data = Y.ptr<float>();

      for (size_t outer = 0; outer < outer_num; ++outer) {
        const float* x_outer_ptr = x_data + outer * x_step;
        float* y_outer_ptr = y_data + outer * y_step;

        for (size_t d = 0; d < dim_size; ++d) {
          const float* src = x_outer_ptr + d * inner_num;
          float* dest = y_outer_ptr + d * multiplier * inner_num;

          for (size_t m = 0; m < multiplier; ++m) { std::copy(src, src + inner_num, dest + m * inner_num); }
        }
      }
      break;
    }
    case kFloat16: {
      const mllm_fp16_t* x_data = X.ptr<mllm_fp16_t>();
      mllm_fp16_t* y_data = Y.ptr<mllm_fp16_t>();

      for (size_t outer = 0; outer < outer_num; ++outer) {
        const mllm_fp16_t* x_outer_ptr = x_data + outer * x_step;
        mllm_fp16_t* y_outer_ptr = y_data + outer * y_step;

        for (size_t d = 0; d < dim_size; ++d) {
          const mllm_fp16_t* src = x_outer_ptr + d * inner_num;
          mllm_fp16_t* dest = y_outer_ptr + d * multiplier * inner_num;

          for (size_t m = 0; m < multiplier; ++m) { std::copy(src, src + inner_num, dest + m * inner_num); }
        }
      }
      break;
    }
    default: NYI("CPURepeatOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
