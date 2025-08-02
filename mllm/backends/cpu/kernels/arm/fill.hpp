// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include <omp.h>

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::arm {
void fill_zeros(mllm_fp32_t* __restrict dst, size_t size, int thread_count);

void fill_ones(mllm_fp32_t* __restrict dst, size_t size, int thread_count);

void fill_specific_value(mllm_fp32_t* __restrict dst, size_t size, mllm_fp32_t value, int thread_count);

void fill_arange(mllm_fp32_t* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step,
                 int thread_count);

void fill_random(mllm_fp32_t* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void fill_zeros_fp16(mllm_fp16_t* __restrict dst, size_t size, int thread_count);

void fill_ones_fp16(mllm_fp16_t* __restrict dst, size_t size, int thread_count);

void fill_specific_value_fp16(mllm_fp16_t* __restrict dst, size_t size, mllm_fp32_t value, int thread_count);

void fill_arange_fp16(mllm_fp16_t* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step,
                      int thread_count);

void fill_random_fp16(mllm_fp16_t* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed,
                      int thread_count);
#endif

// For Normal Int Type
template<typename T>
inline void fill_zeros_anytype(T* __restrict dst, size_t size, int thread_count) {
  if (size == 0) return;

  // If this is a trivial type, use memset. Compiler will optimize this.
  if constexpr (std::is_trivial_v<T>) {
    std::memset(dst, 0, size * sizeof(T));
  } else {
#pragma omp parallel num_threads(thread_count) if (size >= 1024 * 1024 * 4 && thread_count > 1)
    {
      T zero_val{};

      // NOTE: !!!
      // Use static schedule to ensure continuous memory blocks, enhancing cache efficiency.
#pragma omp for schedule(static)
      for (size_t i = 0; i < size; i++) { dst[i] = zero_val; }
    }
  }
}

template<>
inline void fill_zeros_anytype<mllm_fp32_t>(mllm_fp32_t* __restrict dst, size_t size, int thread_count) {
  fill_zeros(dst, size, thread_count);
}

template<>
inline void fill_zeros_anytype<mllm_fp16_t>(mllm_fp16_t* __restrict dst, size_t size, int thread_count) {
  fill_zeros_fp16(dst, size, thread_count);
}

template<typename T>
inline void fill_ones_anytype(T* __restrict dst, size_t size, int thread_count) {
  if (size == 0) return;

#pragma omp parallel for schedule(static) num_threads(thread_count) if (size >= 1024 * 1024 * 4 && thread_count > 1)
  for (size_t i = 0; i < size; ++i) { dst[i] = T(1); }
}

template<>
inline void fill_ones_anytype<mllm_fp32_t>(mllm_fp32_t* __restrict dst, size_t size, int thread_count) {
  fill_ones(dst, size, thread_count);
}

template<>
inline void fill_ones_anytype<mllm_fp16_t>(mllm_fp16_t* __restrict dst, size_t size, int thread_count) {
  fill_ones_fp16(dst, size, thread_count);
}

template<typename T>
inline void fill_specific_value_anytype(T* __restrict dst, size_t size, mllm_fp32_t value, int thread_count) {
#pragma omp parallel for schedule(static) num_threads(thread_count) if (size >= 1024 * 1024 * 4 && thread_count > 1)
  for (size_t i = 0; i < size; ++i) { dst[i] = (T)T(value); }
}

template<>
inline void fill_specific_value_anytype<mllm_fp32_t>(mllm_fp32_t* __restrict dst, size_t size, mllm_fp32_t value,
                                                     int thread_count) {
  fill_specific_value(dst, size, value, thread_count);
}

template<>
inline void fill_specific_value_anytype<mllm_fp16_t>(mllm_fp16_t* __restrict dst, size_t size, mllm_fp32_t value,
                                                     int thread_count) {
  fill_specific_value_fp16(dst, size, value, thread_count);
}

template<typename T>
inline void fill_arange_anytype(T* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step,
                                int thread_count) {
  if (step == 0) {
#pragma omp parallel for num_threads(thread_count) if (size >= 1024 * 1024 * 4 && thread_count > 1)
    for (size_t i = 0; i < size; ++i) { dst[i] = static_cast<T>(start); }
    return;
  }

  size_t n = 0;
  if ((step > 0 && start < end) || (step < 0 && start > end)) {
    mllm_fp32_t n_float = (end - start) / step;
    if (n_float > 0) {
      n = static_cast<size_t>(std::ceil(n_float));
      if (step > 0) {
        if (start + (n - 1) * step >= end) --n;
      } else {
        if (start + (n - 1) * step <= end) --n;
      }
      n = std::min(n, size);
    }
  }

#pragma omp parallel for num_threads(thread_count) if (size >= 1024 * 1024 * 4 && thread_count > 1)
  for (size_t i = 0; i < n; ++i) { dst[i] = static_cast<T>(start + i * step); }
}

template<>
inline void fill_arange_anytype<mllm_fp32_t>(mllm_fp32_t* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end,
                                             mllm_fp32_t step, int thread_count) {
  fill_arange(dst, size, start, end, step, thread_count);
}

template<>
inline void fill_arange_anytype<mllm_fp16_t>(mllm_fp16_t* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end,
                                             mllm_fp32_t step, int thread_count) {
  fill_arange_fp16(dst, size, start, end, step, thread_count);
}

template<typename T>
inline void fill_random_anytype(T* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed,
                                int thread_count) {
  const uint64_t multiplier = 1103515245ULL;
  const uint64_t increment = 12345ULL;
  const uint64_t modulus = 1ULL << 31;  // 2^31
  const mllm_fp32_t range = end - start;

  if (range == 0) {
#pragma omp parallel for num_threads(thread_count) if (size >= 1024 * 128 && thread_count > 1)
    for (size_t i = 0; i < size; ++i) { dst[i] = static_cast<T>(start); }
    return;
  }

#pragma omp parallel num_threads(thread_count) if (size >= 1024 * 128 && thread_count > 1)
  {
    const int tid = omp_get_thread_num();
    const int actual_threads = omp_get_num_threads();
    const size_t chunk_size = (size + actual_threads - 1) / actual_threads;
    const size_t start_idx = tid * chunk_size;
    const size_t end_idx = (tid == actual_threads - 1) ? size : (tid + 1) * chunk_size;

    uint64_t state = seed + static_cast<uint64_t>(tid);
    state = (multiplier * state + increment) % modulus;

    for (size_t i = start_idx; i < end_idx; ++i) {
      state = (multiplier * state + increment) % modulus;
      const mllm_fp32_t random_value = static_cast<mllm_fp32_t>(state) / static_cast<mllm_fp32_t>(modulus - 1);
      dst[i] = static_cast<T>(start + random_value * range);
    }
  }
}

template<>
inline void fill_random_anytype<mllm_fp32_t>(mllm_fp32_t* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end,
                                             uint64_t seed, int thread_count) {
  fill_random(dst, size, start, end, seed, thread_count);
}

template<>
inline void fill_random_anytype<mllm_fp16_t>(mllm_fp16_t* __restrict dst, size_t size, mllm_fp32_t start, mllm_fp32_t end,
                                             uint64_t seed, int thread_count) {
  fill_random_fp16(dst, size, start, end, seed, thread_count);
}

}  // namespace mllm::cpu::arm

#endif