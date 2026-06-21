// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <functional>
#include <vector>
#include <queue>

#include "mllm/backends/cpu/ops/TopKOp.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/UnsafeMacros.hpp"

namespace mllm::cpu {

CPUTopKOp::CPUTopKOp(const aops::TopKOpOptions& options) : aops::TopKOp(options) {}

namespace MLLM_ANONYMOUS_NAMESPACE {

template<typename T>
void topk_impl(const T* input_data, T* values_data, int32_t* indices_data, int outer_size, int axis_size, int inner_size, int k,
               bool largest, bool sorted) {
  std::vector<std::pair<T, int32_t>> heap_data;
  heap_data.reserve(k);

  std::vector<std::pair<T, int32_t>> temp_data;
  temp_data.reserve(k);

  // For largest=true, we want a min-heap so we can efficiently remove the smallest element when we find a larger one
  // For largest=false, we want a max-heap so we can efficiently remove the largest element when we find a smaller one
  auto heap_compare = largest
                          ? [](const std::pair<T, int32_t>& a, const std::pair<T, int32_t>& b) { return a.first > b.first; }
                          : [](const std::pair<T, int32_t>& a, const std::pair<T, int32_t>& b) { return a.first < b.first; };
  auto value_compare = largest
                           ? [](const std::pair<T, int32_t>& a, const std::pair<T, int32_t>& b) { return a.first > b.first; }
                           : [](const std::pair<T, int32_t>& a, const std::pair<T, int32_t>& b) { return a.first < b.first; };
  auto index_compare = [](const std::pair<T, int32_t>& a, const std::pair<T, int32_t>& b) { return a.second < b.second; };

  for (int out = 0; out < outer_size; ++out) {
    for (int in = 0; in < inner_size; ++in) {
      std::priority_queue<std::pair<T, int32_t>, std::vector<std::pair<T, int32_t>>, decltype(heap_compare)> heap(heap_compare);
      for (int i = 0; i < axis_size; ++i) {
        int index = out * axis_size * inner_size + i * inner_size + in;
        std::pair<T, int32_t> item = {input_data[index], static_cast<int32_t>(i)};

        if (heap.size() < k) {
          heap.push(item);
        } else if (heap_compare(item, heap.top())) {
          heap.pop();
          heap.push(item);
        }
      }
      temp_data.clear();
      while (!heap.empty()) {
        temp_data.push_back(heap.top());
        heap.pop();
      }
      if (sorted) {
        std::sort(temp_data.begin(), temp_data.end(), value_compare);
      } else {
        std::sort(temp_data.begin(), temp_data.end(), index_compare);
      }
      for (int i = 0; i < k; ++i) {
        int out_index = out * k * inner_size + i * inner_size + in;
        values_data[out_index] = temp_data[i].first;
        indices_data[out_index] = temp_data[i].second;
      }
    }
  }
}
}  // namespace MLLM_ANONYMOUS_NAMESPACE

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
void CPUTopKOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& values = outputs[0];
  auto& indices = outputs[1];

  auto dtype = input.dtype();
  int k = options_.k;
  int dim = options_.dim;
  bool largest = options_.largest;
  bool sorted = options_.sorted;

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
      topk_impl<float>(input.ptr<float>(), values.ptr<float>(), indices.ptr<int32_t>(), outer_size, axis_size, inner_size, k,
                       largest, sorted);
      break;
    }
    case kFloat16: {
      topk_impl<mllm_fp16_t>(input.ptr<mllm_fp16_t>(), values.ptr<mllm_fp16_t>(), indices.ptr<int32_t>(), outer_size, axis_size,
                             inner_size, k, largest, sorted);
      break;
    }
    case kInt32: {
      topk_impl<int32_t>(input.ptr<int32_t>(), values.ptr<int32_t>(), indices.ptr<int32_t>(), outer_size, axis_size, inner_size,
                         k, largest, sorted);
      break;
    }
    case kInt16: {
      topk_impl<int16_t>(input.ptr<int16_t>(), values.ptr<int16_t>(), indices.ptr<int32_t>(), outer_size, axis_size, inner_size,
                         k, largest, sorted);
      break;
    }
    case kInt8: {
      topk_impl<int8_t>(input.ptr<int8_t>(), values.ptr<int8_t>(), indices.ptr<int32_t>(), outer_size, axis_size, inner_size, k,
                        largest, sorted);
      break;
    }
    default: NYI("Unsupported data type for TopKOp");
  }
}
__MLLM_UNSAFE_OPT_END

}  // namespace mllm::cpu
