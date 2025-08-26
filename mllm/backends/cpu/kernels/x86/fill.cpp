// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/x86/fill.hpp"
#include "mllm/backends/cpu/kernels/x86/simd.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)

namespace mllm::cpu::x86 {

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

void fill_arange(mllm_fp32_t* __restrict dst, size_t size, float start, float end, float step, int thread_count) {
#if defined(MLLM_HOST_FEATURE_AVX512F)
  constexpr size_t vec_size = 16;
  const __m512 step_vec = _mm512_set1_ps(step * vec_size);
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
  constexpr size_t vec_size = 8;
  const __m256 step_vec = _mm256_set1_ps(step * vec_size);
#elif defined(MLLM_HOST_FEATURE_SSE2) || defined(MLLM_HOST_FEATURE_SSE)
  constexpr size_t vec_size = 4;
  const __m128 step_vec = _mm_set1_ps(step * vec_size);
#else
  constexpr size_t vec_size = 1;
#endif

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

#if defined(MLLM_HOST_FEATURE_AVX512F) || defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX) \
    || defined(MLLM_HOST_FEATURE_SSE2) || defined(MLLM_HOST_FEATURE_SSE)
  if (vec_size > 1) {
    // Vectorized arange
    float current_value = start;
    for (; i < vec_end; i += vec_size) {
#if defined(MLLM_HOST_FEATURE_AVX512F)
      __m512 val_vec = _mm512_set_ps(current_value + 15 * step, current_value + 14 * step, current_value + 13 * step,
                                     current_value + 12 * step, current_value + 11 * step, current_value + 10 * step,
                                     current_value + 9 * step, current_value + 8 * step, current_value + 7 * step,
                                     current_value + 6 * step, current_value + 5 * step, current_value + 4 * step,
                                     current_value + 3 * step, current_value + 2 * step, current_value + step, current_value);
      _mm512_storeu_ps(dst + i, val_vec);
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
      __m256 val_vec =
          _mm256_set_ps(current_value + 7 * step, current_value + 6 * step, current_value + 5 * step, current_value + 4 * step,
                        current_value + 3 * step, current_value + 2 * step, current_value + step, current_value);
      _mm256_storeu_ps(dst + i, val_vec);
#elif defined(MLLM_HOST_FEATURE_SSE2) || defined(MLLM_HOST_FEATURE_SSE)
      __m128 val_vec = _mm_set_ps(current_value + 3 * step, current_value + 2 * step, current_value + step, current_value);
      _mm_storeu_ps(dst + i, val_vec);
#endif
      current_value += step * vec_size;
    }
  }
#endif

  // Handle remaining elements
  float current_value = start + i * step;
  for (; i < size; ++i) {
    dst[i] = current_value;
    current_value += step;
  }
}

void fill_random(mllm_fp32_t* __restrict dst, size_t size, float start, float end, uint64_t seed, int thread_count) {
  uint64_t state = seed;
  const uint64_t multiplier = 1103515245ULL;
  const uint64_t increment = 12345ULL;
  const uint64_t modulus = 1ULL << 31;  // 2^31

  float range = end - start;

  for (size_t i = 0; i < size; ++i) {
    state = (multiplier * state + increment) % modulus;

    float random_value = static_cast<float>(state) / static_cast<float>(modulus - 1);
    dst[i] = start + random_value * range;
  }
}

}  // namespace mllm::cpu::x86

#endif
