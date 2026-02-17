// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// NOTE: Do NOT use #pragma once here!
// Highway's foreach_target.h mechanism requires -inl.hpp files to be included
// multiple times, once for each target architecture (AVX3_DL, AVX10_2, etc.).

#include <hwy/highway.h>
#include <cstring>
#include "mllm/core/DataTypes.hpp"

HWY_BEFORE_NAMESPACE();
namespace mllm::cpu::common {  // NOLINT
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

//===----------------------------------------------------------------------===//
// Fill Zeros
//===----------------------------------------------------------------------===//
template<typename T>
HWY_INLINE void fill_zeros_impl(T* HWY_RESTRICT dst, size_t count) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  const hn::Vec<decltype(d)> zero = hn::Zero(d);
  size_t idx = 0;

  for (; idx + N <= count; idx += N) { hn::StoreU(zero, d, dst + idx); }

  if (idx < count) { hn::StoreN(zero, d, dst + idx, count - idx); }
}

// Specialization for types not supported by Highway SIMD, use memset
template<typename T>
HWY_INLINE void fill_zeros_scalar(T* HWY_RESTRICT dst, size_t count) {
  if constexpr (std::is_trivial_v<T>) {
    std::memset(dst, 0, count * sizeof(T));
  } else {
    T zero_val{};
    for (size_t i = 0; i < count; ++i) { dst[i] = zero_val; }
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_fp32(mllm_fp32_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_fp64(mllm_fp64_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_i32(mllm_int32_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_u32(mllm_uint32_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_i64(mllm_int64_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_u64(mllm_uint64_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_i16(mllm_int16_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_u16(mllm_uint16_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_i8(mllm_int8_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_zeros_u8(mllm_uint8_t* HWY_RESTRICT dst, size_t size) {
  fill_zeros_impl(dst, size);
}

//===----------------------------------------------------------------------===//
// Fill Ones
//===----------------------------------------------------------------------===//
template<typename T>
HWY_INLINE void fill_ones_impl(T* HWY_RESTRICT dst, size_t count) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  const hn::Vec<decltype(d)> one = hn::Set(d, static_cast<T>(1));
  size_t idx = 0;

  for (; idx + N <= count; idx += N) { hn::StoreU(one, d, dst + idx); }

  if (idx < count) { hn::StoreN(one, d, dst + idx, count - idx); }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_fp32(mllm_fp32_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_fp64(mllm_fp64_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_i32(mllm_int32_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_u32(mllm_uint32_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_i64(mllm_int64_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_u64(mllm_uint64_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_i16(mllm_int16_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_u16(mllm_uint16_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_i8(mllm_int8_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_ones_u8(mllm_uint8_t* HWY_RESTRICT dst, size_t size) {
  fill_ones_impl(dst, size);
}

//===----------------------------------------------------------------------===//
// Fill Specific Value
//===----------------------------------------------------------------------===//
template<typename T>
HWY_INLINE void fill_value_impl(T* HWY_RESTRICT dst, size_t count, T value) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  const hn::Vec<decltype(d)> v = hn::Set(d, value);
  size_t idx = 0;

  for (; idx + N <= count; idx += N) { hn::StoreU(v, d, dst + idx); }

  if (idx < count) { hn::StoreN(v, d, dst + idx, count - idx); }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_fp32(mllm_fp32_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_fp64(mllm_fp64_t* HWY_RESTRICT dst, size_t size, mllm_fp64_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_i32(mllm_int32_t* HWY_RESTRICT dst, size_t size, mllm_int32_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_u32(mllm_uint32_t* HWY_RESTRICT dst, size_t size, mllm_uint32_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_i64(mllm_int64_t* HWY_RESTRICT dst, size_t size, mllm_int64_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_u64(mllm_uint64_t* HWY_RESTRICT dst, size_t size, mllm_uint64_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_i16(mllm_int16_t* HWY_RESTRICT dst, size_t size, mllm_int16_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_u16(mllm_uint16_t* HWY_RESTRICT dst, size_t size, mllm_uint16_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_i8(mllm_int8_t* HWY_RESTRICT dst, size_t size, mllm_int8_t value) {
  fill_value_impl(dst, size, value);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_value_u8(mllm_uint8_t* HWY_RESTRICT dst, size_t size, mllm_uint8_t value) {
  fill_value_impl(dst, size, value);
}

//===----------------------------------------------------------------------===//
// Fill Arange (start, end, step)
//===----------------------------------------------------------------------===//
template<typename T>
HWY_INLINE void fill_arange_impl(T* HWY_RESTRICT dst, size_t count, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  if (step == 0) {
    fill_value_impl(dst, count, static_cast<T>(start));
    return;
  }

  // Calculate the actual number of elements to fill
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
      n = std::min(n, count);
    }
  }

  // Use SIMD for float types where we can vectorize the computation
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);

    // Create increment vector: [0, 1, 2, 3, ...] * step
    const hn::Vec<decltype(d)> step_vec = hn::Set(d, step);
    const hn::Vec<decltype(d)> n_step_vec = hn::Set(d, step * static_cast<T>(N));

    // Create base offsets [0, 1, 2, 3, ...]
    hn::Vec<decltype(d)> base = hn::Iota(d, 0);
    base = hn::Mul(base, step_vec);
    hn::Vec<decltype(d)> current_start = hn::Add(hn::Set(d, start), base);

    size_t idx = 0;
    for (; idx + N <= n; idx += N) {
      hn::StoreU(current_start, d, dst + idx);
      current_start = hn::Add(current_start, n_step_vec);
    }

    // Handle remaining elements
    for (; idx < n; ++idx) { dst[idx] = static_cast<T>(start + idx * step); }
  } else {
    // Scalar fallback for other types
    for (size_t i = 0; i < n; ++i) { dst[i] = static_cast<T>(start + i * step); }
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_fp32(mllm_fp32_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                           mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_i32(mllm_int32_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_u32(mllm_uint32_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_i64(mllm_int64_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_u64(mllm_uint64_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_i16(mllm_int16_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_u16(mllm_uint16_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_i8(mllm_int8_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                         mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_arange_u8(mllm_uint8_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                         mllm_fp32_t end, mllm_fp32_t step) {
  fill_arange_impl(dst, size, start, end, step);
}

//===----------------------------------------------------------------------===//
// Fill Random (using LCG random number generator)
//===----------------------------------------------------------------------===//
template<typename T>
HWY_INLINE void fill_random_impl(T* HWY_RESTRICT dst, size_t count, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  const uint64_t multiplier = 1103515245ULL;
  const uint64_t increment = 12345ULL;
  const uint64_t modulus = 1ULL << 31;  // 2^31
  const mllm_fp32_t range = end - start;

  if (range == 0) {
    fill_value_impl(dst, count, static_cast<T>(start));
    return;
  }

  uint64_t state = seed;
  state = (multiplier * state + increment) % modulus;

  for (size_t i = 0; i < count; ++i) {
    state = (multiplier * state + increment) % modulus;
    const mllm_fp32_t random_value = static_cast<mllm_fp32_t>(state) / static_cast<mllm_fp32_t>(modulus - 1);
    dst[i] = static_cast<T>(start + random_value * range);
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_fp32(mllm_fp32_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                           mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_i32(mllm_int32_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_u32(mllm_uint32_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_i64(mllm_int64_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_u64(mllm_uint64_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_i16(mllm_int16_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_u16(mllm_uint16_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                          mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_i8(mllm_int8_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                         mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void fill_random_u8(mllm_uint8_t* HWY_RESTRICT dst, size_t size, mllm_fp32_t start,
                                                         mllm_fp32_t end, uint64_t seed) {
  fill_random_impl(dst, size, start, end, seed);
}

}  // namespace HWY_NAMESPACE
}  // namespace mllm::cpu::common
HWY_AFTER_NAMESPACE();
