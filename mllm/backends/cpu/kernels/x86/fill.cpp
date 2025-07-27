/**
 * @file fill.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-27
 *
 */
#include "mllm/backends/cpu/kernels/x86/fill.hpp"
#include "mllm/backends/cpu/kernels/x86/simd.hpp"

namespace mllm::x86 {

void fill_zeros(mllm_fp32_t* __restrict dst, size_t size, int thread_count) {
#if defined(MLLM_HOST_FEATURE_AVX512F)
  constexpr size_t vec_size = 16;  // 16 floats in AVX-512
  const __m512 zero_vec = _mm512_setzero_ps();
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
  constexpr size_t vec_size = 8;  // 8 floats in AVX
  const __m256 zero_vec = _mm256_setzero_ps();
#elif defined(MLLM_HOST_FEATURE_SSE2)
  constexpr size_t vec_size = 4;  // 4 floats in SSE2
  const __m128 zero_vec = _mm_setzero_ps();
#elif defined(MLLM_HOST_FEATURE_SSE)
  constexpr size_t vec_size = 4;  // 4 floats in SSE
  const __m128 zero_vec = _mm_setzero_ps();
#else
  constexpr size_t vec_size = 1;
#endif

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) {
#if defined(MLLM_HOST_FEATURE_AVX512F)
    _mm512_storeu_ps(dst + i, zero_vec);
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
    _mm256_storeu_ps(dst + i, zero_vec);
#elif defined(MLLM_HOST_FEATURE_SSE2) || defined(MLLM_HOST_FEATURE_SSE)
    _mm_storeu_ps(dst + i, zero_vec);
#endif
  }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = 0.0f; }
}

void fill_ones(mllm_fp32_t* __restrict dst, size_t size, int thread_count) {
#if defined(MLLM_HOST_FEATURE_AVX512F)
  constexpr size_t vec_size = 16;
  const __m512 ones_vec = _mm512_set1_ps(1.0f);
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
  constexpr size_t vec_size = 8;
  const __m256 ones_vec = _mm256_set1_ps(1.0f);
#elif defined(MLLM_HOST_FEATURE_SSE2) || defined(MLLM_HOST_FEATURE_SSE)
  constexpr size_t vec_size = 4;
  const __m128 ones_vec = _mm_set1_ps(1.0f);
#else
  constexpr size_t vec_size = 1;
#endif

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) {
#if defined(MLLM_HOST_FEATURE_AVX512F)
    _mm512_storeu_ps(dst + i, ones_vec);
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
    _mm256_storeu_ps(dst + i, ones_vec);
#elif defined(MLLM_HOST_FEATURE_SSE2) || defined(MLLM_HOST_FEATURE_SSE)
    _mm_storeu_ps(dst + i, ones_vec);
#endif
  }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = 1.0f; }
}

void fill_specific_value(mllm_fp32_t* __restrict dst, size_t size, float value, int thread_count) {
#if defined(MLLM_HOST_FEATURE_AVX512F)
  constexpr size_t vec_size = 16;
  const __m512 value_vec = _mm512_set1_ps(value);
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
  constexpr size_t vec_size = 8;
  const __m256 value_vec = _mm256_set1_ps(value);
#elif defined(MLLM_HOST_FEATURE_SSE2) || defined(MLLM_HOST_FEATURE_SSE)
  constexpr size_t vec_size = 4;
  const __m128 value_vec = _mm_set1_ps(value);
#else
  constexpr size_t vec_size = 1;
#endif

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) {
#if defined(MLLM_HOST_FEATURE_AVX512F)
    _mm512_storeu_ps(dst + i, value_vec);
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
    _mm256_storeu_ps(dst + i, value_vec);
#elif defined(MLLM_HOST_FEATURE_SSE2) || defined(MLLM_HOST_FEATURE_SSE)
    _mm_storeu_ps(dst + i, value_vec);
#endif
  }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = value; }
}

}  // namespace mllm::x86
