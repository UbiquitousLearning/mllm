// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

namespace mllm::cpu::common {

inline void compute_strides_internal(const int* shape, int ndim, int* strides) {
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) { strides[i] = strides[i + 1] * shape[i + 1]; }
}

template<typename T>
void permute_generic(const T* __restrict__ input, T* __restrict__ output, const int* __restrict__ in_shape,
                     const int* __restrict__ perm, int ndim) {
  std::vector<int> out_shape(ndim);
  for (int i = 0; i < ndim; ++i) { out_shape[i] = in_shape[perm[i]]; }

  std::vector<int> in_strides(ndim), out_strides(ndim);
  compute_strides_internal(in_shape, ndim, in_strides.data());
  compute_strides_internal(out_shape.data(), ndim, out_strides.data());

  int total_elements = 1;
  for (int i = 0; i < ndim; ++i) { total_elements *= in_shape[i]; }

  // Use simple element-by-element copy for generic types
  std::vector<int> out_coord(ndim);
  std::vector<int> in_coord(ndim);
  for (int i = 0; i < total_elements; ++i) {
    int temp_idx = i;
    for (int d = ndim - 1; d >= 0; --d) {
      out_coord[d] = temp_idx % out_shape[d];
      temp_idx /= out_shape[d];
    }
    for (int d = 0; d < ndim; ++d) { in_coord[perm[d]] = out_coord[d]; }
    int in_offset = 0;
    for (int d = 0; d < ndim; ++d) { in_offset += in_coord[d] * in_strides[d]; }

    output[i] = input[in_offset];
  }
}

}  // namespace mllm::cpu::common
