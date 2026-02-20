// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/x86/transpose.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)

#include <cstddef>
#include <vector>

#if defined(MLLM_HOST_FEATURE_AVX512F) || defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
#include <immintrin.h>
#elif defined(MLLM_HOST_FEATURE_SSE2)
#include <emmintrin.h>
#elif defined(MLLM_HOST_FEATURE_SSE)
#include <xmmintrin.h>
#endif

namespace mllm::cpu::x86 {

namespace {
void compute_strides(const int* shape, int ndim, int* strides) {
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) { strides[i] = strides[i + 1] * shape[i + 1]; }
}
}  // namespace

void transpose_hw_wh_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, size_t H, size_t W) {
#if defined(MLLM_HOST_FEATURE_SSE)
  // Process 4x4 blocks using SSE
  for (size_t i = 0; i + 4 <= H; i += 4) {
    for (size_t j = 0; j + 4 <= W; j += 4) {
      // Load 4 rows, each containing 4 floats
      __m128 r0 = _mm_loadu_ps(X + i * W + j);
      __m128 r1 = _mm_loadu_ps(X + (i + 1) * W + j);
      __m128 r2 = _mm_loadu_ps(X + (i + 2) * W + j);
      __m128 r3 = _mm_loadu_ps(X + (i + 3) * W + j);

      // Transpose 4x4 matrix using SSE intrinsics
      // Step 1: Interleave low and high halves
      __m128 t0 = _mm_unpacklo_ps(r0, r1);  // a0 b0 a1 b1
      __m128 t1 = _mm_unpackhi_ps(r0, r1);  // a2 b2 a3 b3
      __m128 t2 = _mm_unpacklo_ps(r2, r3);  // c0 d0 c1 d1
      __m128 t3 = _mm_unpackhi_ps(r2, r3);  // c2 d2 c3 d3

      // Step 2: Shuffle to get final transposed rows
      __m128 col0 = _mm_movelh_ps(t0, t2);  // a0 b0 c0 d0
      __m128 col1 = _mm_movehl_ps(t2, t0);  // a1 b1 c1 d1
      __m128 col2 = _mm_movelh_ps(t1, t3);  // a2 b2 c2 d2
      __m128 col3 = _mm_movehl_ps(t3, t1);  // a3 b3 c3 d3

      // Store transposed columns as rows in output
      _mm_storeu_ps(Y + j * H + i, col0);
      _mm_storeu_ps(Y + (j + 1) * H + i, col1);
      _mm_storeu_ps(Y + (j + 2) * H + i, col2);
      _mm_storeu_ps(Y + (j + 3) * H + i, col3);
    }

    // Handle remaining columns
    size_t j_remain = W - (W % 4);
    for (size_t j = j_remain; j < W; ++j) {
      __m128 col = _mm_set_ps(X[(i + 3) * W + j], X[(i + 2) * W + j], X[(i + 1) * W + j], X[i * W + j]);
      _mm_storeu_ps(Y + j * H + i, col);
    }
  }

  // Handle remaining rows
  size_t i_remain = H - (H % 4);
  for (size_t j = 0; j < W; ++j) {
    for (size_t i = i_remain; i < H; ++i) { Y[j * H + i] = X[i * W + j]; }
  }
#else
  // Scalar fallback
  for (size_t i = 0; i < H; ++i) {
    for (size_t j = 0; j < W; ++j) { Y[j * H + i] = X[i * W + j]; }
  }
#endif
}

void transpose_bshd_bhsd_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, size_t B, size_t S, size_t H,
                              size_t D) {
#if defined(MLLM_HOST_FEATURE_SSE)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t s = 0; s < S; ++s) {
        size_t d = 0;
        // Process 4 elements at a time using SSE
        for (; d + 4 <= D; d += 4) {
          // B, S, H, D
          const mllm_fp32_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          // B, H, S, D
          mllm_fp32_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;

          __m128 data = _mm_loadu_ps(src_ptr);
          _mm_storeu_ps(dst_ptr, data);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
          const mllm_fp32_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          mllm_fp32_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
#else
  // Scalar fallback
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t s = 0; s < S; ++s) {
        for (size_t d = 0; d < D; ++d) {
          const mllm_fp32_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          mllm_fp32_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
#endif
}

void transpose_last_dims_fp32(const mllm_fp32_t* __restrict input, mllm_fp32_t* __restrict output, size_t batch, size_t dim0,
                              size_t dim1) {
#if defined(MLLM_HOST_FEATURE_SSE)
  for (size_t b = 0; b < batch; b++) {
    const mllm_fp32_t* input_batch = input + b * dim0 * dim1;
    mllm_fp32_t* output_batch = output + b * dim0 * dim1;

    // Process 4x4 blocks
    for (size_t i = 0; i + 4 <= dim0; i += 4) {
      for (size_t j = 0; j + 4 <= dim1; j += 4) {
        __m128 r0 = _mm_loadu_ps(input_batch + i * dim1 + j);
        __m128 r1 = _mm_loadu_ps(input_batch + (i + 1) * dim1 + j);
        __m128 r2 = _mm_loadu_ps(input_batch + (i + 2) * dim1 + j);
        __m128 r3 = _mm_loadu_ps(input_batch + (i + 3) * dim1 + j);

        // Transpose 4x4 matrix
        __m128 t0 = _mm_unpacklo_ps(r0, r1);
        __m128 t1 = _mm_unpackhi_ps(r0, r1);
        __m128 t2 = _mm_unpacklo_ps(r2, r3);
        __m128 t3 = _mm_unpackhi_ps(r2, r3);

        __m128 col0 = _mm_movelh_ps(t0, t2);
        __m128 col1 = _mm_movehl_ps(t2, t0);
        __m128 col2 = _mm_movelh_ps(t1, t3);
        __m128 col3 = _mm_movehl_ps(t3, t1);

        _mm_storeu_ps(output_batch + j * dim0 + i, col0);
        _mm_storeu_ps(output_batch + (j + 1) * dim0 + i, col1);
        _mm_storeu_ps(output_batch + (j + 2) * dim0 + i, col2);
        _mm_storeu_ps(output_batch + (j + 3) * dim0 + i, col3);
      }

      // Handle remaining columns in the block
      size_t j_remain = dim1 - (dim1 % 4);
      for (size_t j = j_remain; j < dim1; ++j) {
        __m128 col = _mm_set_ps(input_batch[(i + 3) * dim1 + j], input_batch[(i + 2) * dim1 + j],
                                input_batch[(i + 1) * dim1 + j], input_batch[i * dim1 + j]);
        _mm_storeu_ps(output_batch + j * dim0 + i, col);
      }
    }

    // Handle remaining rows
    size_t i_remain = dim0 - (dim0 % 4);
    for (size_t j = 0; j < dim1; ++j) {
      for (size_t i = i_remain; i < dim0; ++i) { output_batch[j * dim0 + i] = input_batch[i * dim1 + j]; }
    }
  }
#else
  // Scalar fallback
  for (size_t b = 0; b < batch; b++) {
    const mllm_fp32_t* input_batch = input + b * dim0 * dim1;
    mllm_fp32_t* output_batch = output + b * dim0 * dim1;
    for (size_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < dim1; ++j) { output_batch[j * dim0 + i] = input_batch[i * dim1 + j]; }
    }
  }
#endif
}

void transpose_hw_wh_int64(const mllm_int64_t* __restrict X, mllm_int64_t* __restrict Y, size_t H, size_t W) {
#if defined(MLLM_HOST_FEATURE_SSE2)
  // Process 2x2 blocks using SSE2 (128-bit registers hold 2 int64)
  for (size_t i = 0; i + 2 <= H; i += 2) {
    for (size_t j = 0; j + 2 <= W; j += 2) {
      // Load 2 rows, each containing 2 int64s
      __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(X + i * W + j));
      __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(X + (i + 1) * W + j));

      // Transpose 2x2 matrix
      // col0 = [r0[0], r1[0]]
      // col1 = [r0[1], r1[1]]
      __m128i col0 = _mm_unpacklo_epi64(r0, r1);
      __m128i col1 = _mm_unpackhi_epi64(r0, r1);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(Y + j * H + i), col0);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(Y + (j + 1) * H + i), col1);
    }

    // Handle remaining columns
    size_t j_remain = W - (W % 2);
    for (size_t j = j_remain; j < W; ++j) {
      __m128i col = _mm_set_epi64x(X[(i + 1) * W + j], X[i * W + j]);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(Y + j * H + i), col);
    }
  }

  // Handle remaining rows
  size_t i_remain = H - (H % 2);
  for (size_t j = 0; j < W; ++j) {
    for (size_t i = i_remain; i < H; ++i) { Y[j * H + i] = X[i * W + j]; }
  }
#else
  // Scalar fallback
  for (size_t i = 0; i < H; ++i) {
    for (size_t j = 0; j < W; ++j) { Y[j * H + i] = X[i * W + j]; }
  }
#endif
}

void transpose_bshd_bhsd_int64(const mllm_int64_t* __restrict X, mllm_int64_t* __restrict Y, size_t B, size_t S, size_t H,
                               size_t D) {
#if defined(MLLM_HOST_FEATURE_SSE2)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t s = 0; s < S; ++s) {
        size_t d = 0;
        // Process 2 elements at a time using SSE2
        for (; d + 2 <= D; d += 2) {
          // B, S, H, D
          const mllm_int64_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          // B, H, S, D
          mllm_int64_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;

          __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr), data);
        }
        // Handle remaining element
        for (; d < D; ++d) {
          const mllm_int64_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          mllm_int64_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
#else
  // Scalar fallback
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t s = 0; s < S; ++s) {
        for (size_t d = 0; d < D; ++d) {
          const mllm_int64_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          mllm_int64_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
#endif
}

void transpose_last_dims_int64(const mllm_int64_t* __restrict input, mllm_int64_t* __restrict output, size_t batch, size_t dim0,
                               size_t dim1) {
#if defined(MLLM_HOST_FEATURE_SSE2)
  for (size_t b = 0; b < batch; b++) {
    const mllm_int64_t* input_batch = input + b * dim0 * dim1;
    mllm_int64_t* output_batch = output + b * dim0 * dim1;

    // Process 2x2 blocks
    for (size_t i = 0; i + 2 <= dim0; i += 2) {
      for (size_t j = 0; j + 2 <= dim1; j += 2) {
        __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_batch + i * dim1 + j));
        __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_batch + (i + 1) * dim1 + j));

        __m128i col0 = _mm_unpacklo_epi64(r0, r1);
        __m128i col1 = _mm_unpackhi_epi64(r0, r1);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_batch + j * dim0 + i), col0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_batch + (j + 1) * dim0 + i), col1);
      }

      // Handle remaining columns
      size_t j_remain = dim1 - (dim1 % 2);
      for (size_t j = j_remain; j < dim1; ++j) {
        __m128i col = _mm_set_epi64x(input_batch[(i + 1) * dim1 + j], input_batch[i * dim1 + j]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_batch + j * dim0 + i), col);
      }
    }

    // Handle remaining rows
    size_t i_remain = dim0 - (dim0 % 2);
    for (size_t j = 0; j < dim1; ++j) {
      for (size_t i = i_remain; i < dim0; ++i) { output_batch[j * dim0 + i] = input_batch[i * dim1 + j]; }
    }
  }
#else
  // Scalar fallback
  for (size_t b = 0; b < batch; b++) {
    const mllm_int64_t* input_batch = input + b * dim0 * dim1;
    mllm_int64_t* output_batch = output + b * dim0 * dim1;
    for (size_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < dim1; ++j) { output_batch[j * dim0 + i] = input_batch[i * dim1 + j]; }
    }
  }
#endif
}

void permute_fp32(const mllm_fp32_t* __restrict input, mllm_fp32_t* __restrict output, const int* __restrict in_shape,
                  const int* __restrict perm, int ndim) {
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
#if defined(MLLM_HOST_FEATURE_SSE)
    const int chunk_size = 4;
#else
    const int chunk_size = 1;
#endif
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
#if defined(MLLM_HOST_FEATURE_SSE)
      for (; j <= inner_dim_size - chunk_size; j += chunk_size) {
        __m128 vec = _mm_loadu_ps(in_ptr + j);
        _mm_storeu_ps(out_ptr + j, vec);
      }
#endif
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

template<typename T>
void permute_generic(const T* __restrict input, T* __restrict output, const int* __restrict in_shape,
                     const int* __restrict perm, int ndim) {
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
template void permute_generic<mllm_int8_t>(const mllm_int8_t* __restrict input, mllm_int8_t* __restrict output,
                                           const int* __restrict in_shape, const int* __restrict perm, int ndim);
template void permute_generic<mllm_uint8_t>(const mllm_uint8_t* __restrict input, mllm_uint8_t* __restrict output,
                                            const int* __restrict in_shape, const int* __restrict perm, int ndim);
template void permute_generic<mllm_int16_t>(const mllm_int16_t* __restrict input, mllm_int16_t* __restrict output,
                                            const int* __restrict in_shape, const int* __restrict perm, int ndim);
template void permute_generic<mllm_uint16_t>(const mllm_uint16_t* __restrict input, mllm_uint16_t* __restrict output,
                                             const int* __restrict in_shape, const int* __restrict perm, int ndim);
template void permute_generic<mllm_int32_t>(const mllm_int32_t* __restrict input, mllm_int32_t* __restrict output,
                                            const int* __restrict in_shape, const int* __restrict perm, int ndim);
template void permute_generic<mllm_uint32_t>(const mllm_uint32_t* __restrict input, mllm_uint32_t* __restrict output,
                                             const int* __restrict in_shape, const int* __restrict perm, int ndim);
template void permute_generic<mllm_int64_t>(const mllm_int64_t* __restrict input, mllm_int64_t* __restrict output,
                                            const int* __restrict in_shape, const int* __restrict perm, int ndim);
template void permute_generic<mllm_uint64_t>(const mllm_uint64_t* __restrict input, mllm_uint64_t* __restrict output,
                                             const int* __restrict in_shape, const int* __restrict perm, int ndim);

}  // namespace mllm::cpu::x86

#endif
