// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <functional>
#include <vector>

#include "mllm/backends/cpu/ops/ArgsortOp.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/UnsafeMacros.hpp"

namespace mllm::cpu {

CPUArgsortOp::CPUArgsortOp(const aops::ArgsortOpOptions& options) : aops::ArgsortOp(options) {}

namespace MLLM_ANONYMOUS_NAMESPACE {

template<typename T>
void argsort_impl(const T* input_data, int32_t* indices_data, int outer_size, int axis_size, int inner_size, int dim,
                  bool descending) {
  for (int out = 0; out < outer_size; ++out) {
    for (int in = 0; in < inner_size; ++in) {
      // Create pairs of (value, index) for sorting
      std::vector<std::pair<T, int32_t>> data_pairs(axis_size);
      for (int i = 0; i < axis_size; ++i) {
        int index = out * axis_size * inner_size + i * inner_size + in;
        data_pairs[i] = {input_data[index], static_cast<int32_t>(i)};
      }

      // Sort based on values
      if (descending) {
        std::sort(data_pairs.begin(), data_pairs.end(),
                  [](const std::pair<T, int32_t>& a, const std::pair<T, int32_t>& b) { return a.first > b.first; });
      } else {
        std::sort(data_pairs.begin(), data_pairs.end(),
                  [](const std::pair<T, int32_t>& a, const std::pair<T, int32_t>& b) { return a.first < b.first; });
      }

      // Store sorted indices
      for (int i = 0; i < axis_size; ++i) {
        int out_index = out * axis_size * inner_size + i * inner_size + in;
        indices_data[out_index] = data_pairs[i].second;
      }
    }
  }
}
}  // namespace MLLM_ANONYMOUS_NAMESPACE

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
void CPUArgsortOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& indices = outputs[0];

  auto dtype = input.dtype();
  int dim = options_.dim;
  bool descending = options_.descending;

  // Handle negative dimension index
  if (dim < 0) { dim += input.shape().size(); }

  // Calculate sizes
  int outer_size = 1;
  int inner_size = 1;
  int axis_size = input.shape()[dim];

  for (int i = 0; i < dim; ++i) { outer_size *= input.shape()[i]; }
  for (int i = dim + 1; i < input.shape().size(); ++i) { inner_size *= input.shape()[i]; }

  switch (dtype) {
    case kFloat32: {
      argsort_impl<float>(input.ptr<float>(), indices.ptr<mllm_int32_t>(), outer_size, axis_size, inner_size, dim, descending);
      break;
    }
    case kFloat16: {
      argsort_impl<mllm_fp16_t>(input.ptr<mllm_fp16_t>(), indices.ptr<mllm_int32_t>(), outer_size, axis_size, inner_size, dim,
                                descending);
      break;
    }
    case kInt32: {
      argsort_impl<mllm_int32_t>(input.ptr<mllm_int32_t>(), indices.ptr<mllm_int32_t>(), outer_size, axis_size, inner_size, dim,
                                 descending);
      break;
    }
    case kInt16: {
      argsort_impl<mllm_int16_t>(input.ptr<mllm_int16_t>(), indices.ptr<mllm_int32_t>(), outer_size, axis_size, inner_size, dim,
                                 descending);
      break;
    }
    case kInt8: {
      argsort_impl<mllm_int8_t>(input.ptr<mllm_int8_t>(), indices.ptr<mllm_int32_t>(), outer_size, axis_size, inner_size, dim,
                                descending);
      break;
    }
    default: NYI("Unsupported data type for ArgsortOp");
  }
}
__MLLM_UNSAFE_OPT_END

}  // namespace mllm::cpu
