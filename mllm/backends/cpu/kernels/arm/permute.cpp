// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/permute.hpp"
#include <vector>
#include "mllm/utils/Common.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::arm {

namespace MLLM_ANONYMOUS_NAMESPACE {
void compute_strides(const int* shape, int ndim, int* strides) {
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) { strides[i] = strides[i + 1] * shape[i + 1]; }
}
}  // namespace MLLM_ANONYMOUS_NAMESPACE

void permute_fp32(const mllm_fp32_t* __restrict__ input, mllm_fp32_t* __restrict__ output, const int* __restrict__ in_shape,
                  const int* __restrict__ perm, int ndim) {
  std::vector<int> out_shape(ndim);
  for (int i = 0; i < ndim; ++i) { out_shape[i] = in_shape[perm[i]]; }
  std::vector<int> in_strides(ndim), out_strides(ndim);
  compute_strides(in_shape, ndim, in_strides.data());
  compute_strides(out_shape.data(), ndim, out_strides.data());
  int total_elements = 1;
  for (int i = 0; i < ndim; ++i) { total_elements *= in_shape[i]; }
  bool inner_dim_contiguous = (perm[ndim - 1] == ndim - 1);
  int inner_dim_size = out_shape[ndim - 1];
  if (inner_dim_contiguous && inner_dim_size >= 4) {
    int outer_elements = total_elements / inner_dim_size;
    int chunk_size = 4;
    for (int outer_idx = 0; outer_idx < outer_elements; ++outer_idx) {
      std::vector<int> coord(ndim - 1);
      int temp = outer_idx;
      for (int i = ndim - 2; i >= 0; --i) {
        coord[i] = temp % out_shape[i];
        temp /= out_shape[i];
      }
      int in_offset = 0;
      int out_offset = 0;
      for (int i = 0; i < ndim - 1; ++i) {
        int orig_dim = perm[i];
        in_offset += coord[i] * in_strides[orig_dim];
        out_offset += coord[i] * out_strides[i];
      }
      const float* in_ptr = input + in_offset;
      float* out_ptr = output + out_offset;
      int j = 0;
      for (; j <= inner_dim_size - chunk_size; j += chunk_size) {
        float32x4_t vec = vld1q_f32(in_ptr + j);
        vst1q_f32(out_ptr + j, vec);
      }
      for (; j < inner_dim_size; ++j) { out_ptr[j] = in_ptr[j]; }
    }
  } else {
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
}

void permute_fp16(const mllm_fp16_t* __restrict__ input, mllm_fp16_t* __restrict__ output, const int* __restrict__ in_shape,
                  const int* __restrict__ perm, int ndim) {
  std::vector<int> out_shape(ndim);
  for (int i = 0; i < ndim; ++i) { out_shape[i] = in_shape[perm[i]]; }

  std::vector<int> in_strides(ndim), out_strides(ndim);
  compute_strides(in_shape, ndim, in_strides.data());
  compute_strides(out_shape.data(), ndim, out_strides.data());

  int total_elements = 1;
  for (int i = 0; i < ndim; ++i) { total_elements *= in_shape[i]; }

  bool inner_dim_contiguous = (perm[ndim - 1] == ndim - 1);
  int inner_dim_size = out_shape[ndim - 1];

  if (inner_dim_contiguous && inner_dim_size >= 8) {
    int outer_elements = total_elements / inner_dim_size;
    const int chunk_size = 8;

    for (int outer_idx = 0; outer_idx < outer_elements; ++outer_idx) {
      std::vector<int> coord(ndim - 1);
      int temp = outer_idx;
      for (int i = ndim - 2; i >= 0; --i) {
        coord[i] = temp % out_shape[i];
        temp /= out_shape[i];
      }

      int in_offset = 0;
      int out_offset = 0;
      for (int i = 0; i < ndim - 1; ++i) {
        int orig_dim = perm[i];
        in_offset += coord[i] * in_strides[orig_dim];
        out_offset += coord[i] * out_strides[i];
      }

      const float16_t* in_ptr = input + in_offset;
      float16_t* out_ptr = output + out_offset;

      int j = 0;
      for (; j <= inner_dim_size - chunk_size; j += chunk_size) {
        float16x8_t vec = vld1q_f16(in_ptr + j);
        vst1q_f16(out_ptr + j, vec);
      }

      if (inner_dim_size - j >= 4) {
        float16x4_t vec4 = vld1_f16(in_ptr + j);
        vst1_f16(out_ptr + j, vec4);
        j += 4;
      }

      for (; j < inner_dim_size; j++) { out_ptr[j] = in_ptr[j]; }
    }
  } else {
    for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
      std::vector<int> out_coord(ndim);
      int temp = linear_idx;
      for (int i = ndim - 1; i >= 0; --i) {
        out_coord[i] = temp % out_shape[i];
        temp /= out_shape[i];
      }

      std::vector<int> in_coord(ndim);
      for (int i = 0; i < ndim; ++i) { in_coord[perm[i]] = out_coord[i]; }

      int in_idx = 0;
      for (int i = 0; i < ndim; ++i) { in_idx += in_coord[i] * in_strides[i]; }

      output[linear_idx] = input[in_idx];
    }
  }
}

template<typename T>
void permute_generic(const T* __restrict__ input, T* __restrict__ output, const int* __restrict__ in_shape,
                     const int* __restrict__ perm, int ndim) {
  std::vector<int> out_shape(ndim);
  for (int i = 0; i < ndim; ++i) { out_shape[i] = in_shape[perm[i]]; }

  std::vector<int> in_strides(ndim), out_strides(ndim);
  compute_strides(in_shape, ndim, in_strides.data());
  compute_strides(out_shape.data(), ndim, out_strides.data());

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

// Explicit template instantiations for commonly used types
template void permute_generic<mllm_int8_t>(const mllm_int8_t* __restrict__ input, mllm_int8_t* __restrict__ output,
                                           const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);
template void permute_generic<mllm_uint8_t>(const mllm_uint8_t* __restrict__ input, mllm_uint8_t* __restrict__ output,
                                            const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);
template void permute_generic<mllm_int16_t>(const mllm_int16_t* __restrict__ input, mllm_int16_t* __restrict__ output,
                                            const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);
template void permute_generic<mllm_uint16_t>(const mllm_uint16_t* __restrict__ input, mllm_uint16_t* __restrict__ output,
                                             const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);
template void permute_generic<mllm_int32_t>(const mllm_int32_t* __restrict__ input, mllm_int32_t* __restrict__ output,
                                            const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);
template void permute_generic<mllm_uint32_t>(const mllm_uint32_t* __restrict__ input, mllm_uint32_t* __restrict__ output,
                                             const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);
template void permute_generic<mllm_int64_t>(const mllm_int64_t* __restrict__ input, mllm_int64_t* __restrict__ output,
                                            const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);
template void permute_generic<mllm_uint64_t>(const mllm_uint64_t* __restrict__ input, mllm_uint64_t* __restrict__ output,
                                             const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim);

}  // namespace mllm::cpu::arm

#endif
