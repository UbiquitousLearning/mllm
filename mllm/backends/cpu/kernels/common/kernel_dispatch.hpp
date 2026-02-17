// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#ifndef MLLM_BACKENDS_CPU_KERNELS_COMMON_KERNEL_DISPATCH_HPP_
#define MLLM_BACKENDS_CPU_KERNELS_COMMON_KERNEL_DISPATCH_HPP_

#include "mllm/utils/CPUArchHelper.hpp"
#if !(defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM))

#include <cstring>
#include "mllm/core/DataTypes.hpp"

// Platform-specific definitions used for declaring an interface, independent of
// the SIMD instruction set.
#include <hwy/base.h>  // HWY_RESTRICT
namespace mllm::cpu::common {

//===----------------------------------------------------------------------===//
// Elementwise + - * / By Matrix
//===----------------------------------------------------------------------===//
/// @brief Elementwise operations on contiguous buffers: out[i] = x[i] (op) y[i].
/// @param out Output buffer of length n.
/// @param x Input buffer of length n.
/// @param y Input buffer of length n.
/// @param n Number of elements.
/// @note For integer division, behavior is undefined when a divisor is zero.
HWY_DLLEXPORT void call_elewise_add_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_div_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n);
//TODO: fp16 support not implemented yet
// HWY_DLLEXPORT void call_elewise_add_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t* y, size_t n);
// HWY_DLLEXPORT void call_elewise_sub_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t* y, size_t n);
// HWY_DLLEXPORT void call_elewise_mul_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t* y, size_t n);
// HWY_DLLEXPORT void call_elewise_div_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_add_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_div_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_add_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t* y, size_t n);
// HWY_DLLEXPORT void call_elewise_div_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_add_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t* y, size_t n);
// HWY_DLLEXPORT void call_elewise_div_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t* y, size_t n);

//===----------------------------------------------------------------------===//
// Elementwise + - * / By Const
//===----------------------------------------------------------------------===//
/// @brief Elementwise operations with a scalar constant: out[i] = x[i] (op) y.
/// @param out Output buffer of length n.
/// @param x Input buffer of length n.
/// @param y Scalar constant.
/// @param n Number of elements.
/// @note For integer division, behavior is undefined when y == 0.
HWY_DLLEXPORT void call_elewise_add_scl_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_scl_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_scl_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_div_scl_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n);
//TODO: fp16 support not implemented yet
// HWY_DLLEXPORT void call_elewise_add_scl_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, mllm_fp16_t y, size_t n);
// HWY_DLLEXPORT void call_elewise_sub_scl_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, mllm_fp16_t y, size_t n);
// HWY_DLLEXPORT void call_elewise_mul_scl_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, mllm_fp16_t y, size_t n);
// HWY_DLLEXPORT void call_elewise_div_scl_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, mllm_fp16_t y, size_t n);
HWY_DLLEXPORT void call_elewise_add_scl_int32(mllm_int32_t* out, const mllm_int32_t* x, mllm_int32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_scl_int32(mllm_int32_t* out, const mllm_int32_t* x, mllm_int32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_scl_int32(mllm_int32_t* out, const mllm_int32_t* x, mllm_int32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_div_scl_int32(mllm_int32_t* out, const mllm_int32_t* x, mllm_int32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_add_scl_int16(mllm_int16_t* out, const mllm_int16_t* x, mllm_int16_t y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_scl_int16(mllm_int16_t* out, const mllm_int16_t* x, mllm_int16_t y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_scl_int16(mllm_int16_t* out, const mllm_int16_t* x, mllm_int16_t y, size_t n);
HWY_DLLEXPORT void call_elewise_div_scl_int16(mllm_int16_t* out, const mllm_int16_t* x, mllm_int16_t y, size_t n);
HWY_DLLEXPORT void call_elewise_add_scl_int8(mllm_int8_t* out, const mllm_int8_t* x, mllm_int8_t y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_scl_int8(mllm_int8_t* out, const mllm_int8_t* x, mllm_int8_t y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_scl_int8(mllm_int8_t* out, const mllm_int8_t* x, mllm_int8_t y, size_t n);
HWY_DLLEXPORT void call_elewise_div_scl_int8(mllm_int8_t* out, const mllm_int8_t* x, mllm_int8_t y, size_t n);

//===----------------------------------------------------------------------===//
// Template wrapper for generic elewise operations
//===----------------------------------------------------------------------===//
template<typename T>
inline void elewise_add_anytype(T* out, const T* x, const T* y, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_elewise_add_fp32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_elewise_add_int32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_elewise_add_int16(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_elewise_add_int8(out, x, y, n);
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { out[i] = x[i] + y[i]; }
  }
}

template<typename T>
inline void elewise_sub_anytype(T* out, const T* x, const T* y, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_elewise_sub_fp32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_elewise_sub_int32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_elewise_sub_int16(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_elewise_sub_int8(out, x, y, n);
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { out[i] = x[i] - y[i]; }
  }
}

template<typename T>
inline void elewise_mul_anytype(T* out, const T* x, const T* y, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_elewise_mul_fp32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_elewise_mul_int32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_elewise_mul_int16(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_elewise_mul_int8(out, x, y, n);
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { out[i] = x[i] * y[i]; }
  }
}

template<typename T>
inline void elewise_div_anytype(T* out, const T* x, const T* y, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_elewise_div_fp32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_elewise_div_int32(out, x, y, n);
  } else {
    // Fallback (note: division by zero is undefined)
    for (size_t i = 0; i < n; ++i) { out[i] = x[i] / y[i]; }
  }
}

template<typename T>
inline void elewise_add_scl_anytype(T* out, const T* x, T y, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_elewise_add_scl_fp32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_elewise_add_scl_int32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_elewise_add_scl_int16(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_elewise_add_scl_int8(out, x, y, n);
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { out[i] = x[i] + y; }
  }
}

template<typename T>
inline void elewise_sub_scl_anytype(T* out, const T* x, T y, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_elewise_sub_scl_fp32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_elewise_sub_scl_int32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_elewise_sub_scl_int16(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_elewise_sub_scl_int8(out, x, y, n);
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { out[i] = x[i] - y; }
  }
}

template<typename T>
inline void elewise_mul_scl_anytype(T* out, const T* x, T y, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_elewise_mul_scl_fp32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_elewise_mul_scl_int32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_elewise_mul_scl_int16(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_elewise_mul_scl_int8(out, x, y, n);
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { out[i] = x[i] * y; }
  }
}

template<typename T>
inline void elewise_div_scl_anytype(T* out, const T* x, T y, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_elewise_div_scl_fp32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_elewise_div_scl_int32(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_elewise_div_scl_int16(out, x, y, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_elewise_div_scl_int8(out, x, y, n);
  } else {
    // Fallback (note: division by zero is undefined)
    for (size_t i = 0; i < n; ++i) { out[i] = x[i] / y; }
  }
}

//===----------------------------------------------------------------------===//
// Fill Zeros
//===----------------------------------------------------------------------===//
HWY_DLLEXPORT void call_fill_zeros_fp32(mllm_fp32_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_fp64(mllm_fp64_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_i32(mllm_int32_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_u32(mllm_uint32_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_i64(mllm_int64_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_u64(mllm_uint64_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_i16(mllm_int16_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_u16(mllm_uint16_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_i8(mllm_int8_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_zeros_u8(mllm_uint8_t* dst, size_t n);

//===----------------------------------------------------------------------===//
// Fill Ones
//===----------------------------------------------------------------------===//
HWY_DLLEXPORT void call_fill_ones_fp32(mllm_fp32_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_fp64(mllm_fp64_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_i32(mllm_int32_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_u32(mllm_uint32_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_i64(mllm_int64_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_u64(mllm_uint64_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_i16(mllm_int16_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_u16(mllm_uint16_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_i8(mllm_int8_t* dst, size_t n);
HWY_DLLEXPORT void call_fill_ones_u8(mllm_uint8_t* dst, size_t n);

//===----------------------------------------------------------------------===//
// Fill Specific Value
//===----------------------------------------------------------------------===//
HWY_DLLEXPORT void call_fill_value_fp32(mllm_fp32_t* dst, size_t n, mllm_fp32_t value);
HWY_DLLEXPORT void call_fill_value_fp64(mllm_fp64_t* dst, size_t n, mllm_fp64_t value);
HWY_DLLEXPORT void call_fill_value_i32(mllm_int32_t* dst, size_t n, mllm_int32_t value);
HWY_DLLEXPORT void call_fill_value_u32(mllm_uint32_t* dst, size_t n, mllm_uint32_t value);
HWY_DLLEXPORT void call_fill_value_i64(mllm_int64_t* dst, size_t n, mllm_int64_t value);
HWY_DLLEXPORT void call_fill_value_u64(mllm_uint64_t* dst, size_t n, mllm_uint64_t value);
HWY_DLLEXPORT void call_fill_value_i16(mllm_int16_t* dst, size_t n, mllm_int16_t value);
HWY_DLLEXPORT void call_fill_value_u16(mllm_uint16_t* dst, size_t n, mllm_uint16_t value);
HWY_DLLEXPORT void call_fill_value_i8(mllm_int8_t* dst, size_t n, mllm_int8_t value);
HWY_DLLEXPORT void call_fill_value_u8(mllm_uint8_t* dst, size_t n, mllm_uint8_t value);

//===----------------------------------------------------------------------===//
// Fill Arange
//===----------------------------------------------------------------------===//
HWY_DLLEXPORT void call_fill_arange_fp32(mllm_fp32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);
HWY_DLLEXPORT void call_fill_arange_i32(mllm_int32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);
HWY_DLLEXPORT void call_fill_arange_u32(mllm_uint32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);
HWY_DLLEXPORT void call_fill_arange_i64(mllm_int64_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);
HWY_DLLEXPORT void call_fill_arange_u64(mllm_uint64_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);
HWY_DLLEXPORT void call_fill_arange_i16(mllm_int16_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);
HWY_DLLEXPORT void call_fill_arange_u16(mllm_uint16_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);
HWY_DLLEXPORT void call_fill_arange_i8(mllm_int8_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);
HWY_DLLEXPORT void call_fill_arange_u8(mllm_uint8_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step);

//===----------------------------------------------------------------------===//
// Fill Random
//===----------------------------------------------------------------------===//
HWY_DLLEXPORT void call_fill_random_fp32(mllm_fp32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);
HWY_DLLEXPORT void call_fill_random_i32(mllm_int32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);
HWY_DLLEXPORT void call_fill_random_u32(mllm_uint32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);
HWY_DLLEXPORT void call_fill_random_i64(mllm_int64_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);
HWY_DLLEXPORT void call_fill_random_u64(mllm_uint64_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);
HWY_DLLEXPORT void call_fill_random_i16(mllm_int16_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);
HWY_DLLEXPORT void call_fill_random_u16(mllm_uint16_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);
HWY_DLLEXPORT void call_fill_random_i8(mllm_int8_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);
HWY_DLLEXPORT void call_fill_random_u8(mllm_uint8_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed);

//===----------------------------------------------------------------------===//
// Template wrapper for generic fill operations
//===----------------------------------------------------------------------===//
template<typename T>
inline void fill_zeros_anytype(T* dst, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_fill_zeros_fp32(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_fp64_t>) {
    call_fill_zeros_fp64(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_fill_zeros_i32(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_uint32_t>) {
    call_fill_zeros_u32(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_int64_t>) {
    call_fill_zeros_i64(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_uint64_t>) {
    call_fill_zeros_u64(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_fill_zeros_i16(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_uint16_t>) {
    call_fill_zeros_u16(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_fill_zeros_i8(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_uint8_t>) {
    call_fill_zeros_u8(dst, n);
  } else {
    // Fallback for unsupported types
    std::memset(dst, 0, n * sizeof(T));
  }
}

template<typename T>
inline void fill_ones_anytype(T* dst, size_t n) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_fill_ones_fp32(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_fp64_t>) {
    call_fill_ones_fp64(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_fill_ones_i32(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_uint32_t>) {
    call_fill_ones_u32(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_int64_t>) {
    call_fill_ones_i64(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_uint64_t>) {
    call_fill_ones_u64(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_fill_ones_i16(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_uint16_t>) {
    call_fill_ones_u16(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_fill_ones_i8(dst, n);
  } else if constexpr (std::is_same_v<T, mllm_uint8_t>) {
    call_fill_ones_u8(dst, n);
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { dst[i] = static_cast<T>(1); }
  }
}

template<typename T>
inline void fill_value_anytype(T* dst, size_t n, mllm_fp32_t value) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_fill_value_fp32(dst, n, value);
  } else if constexpr (std::is_same_v<T, mllm_fp64_t>) {
    call_fill_value_fp64(dst, n, static_cast<mllm_fp64_t>(value));
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_fill_value_i32(dst, n, static_cast<mllm_int32_t>(value));
  } else if constexpr (std::is_same_v<T, mllm_uint32_t>) {
    call_fill_value_u32(dst, n, static_cast<mllm_uint32_t>(value));
  } else if constexpr (std::is_same_v<T, mllm_int64_t>) {
    call_fill_value_i64(dst, n, static_cast<mllm_int64_t>(value));
  } else if constexpr (std::is_same_v<T, mllm_uint64_t>) {
    call_fill_value_u64(dst, n, static_cast<mllm_uint64_t>(value));
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_fill_value_i16(dst, n, static_cast<mllm_int16_t>(value));
  } else if constexpr (std::is_same_v<T, mllm_uint16_t>) {
    call_fill_value_u16(dst, n, static_cast<mllm_uint16_t>(value));
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_fill_value_i8(dst, n, static_cast<mllm_int8_t>(value));
  } else if constexpr (std::is_same_v<T, mllm_uint8_t>) {
    call_fill_value_u8(dst, n, static_cast<mllm_uint8_t>(value));
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { dst[i] = static_cast<T>(value); }
  }
}

template<typename T>
inline void fill_arange_anytype(T* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_fill_arange_fp32(dst, n, start, end, step);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_fill_arange_i32(dst, n, start, end, step);
  } else if constexpr (std::is_same_v<T, mllm_uint32_t>) {
    call_fill_arange_u32(dst, n, start, end, step);
  } else if constexpr (std::is_same_v<T, mllm_int64_t>) {
    call_fill_arange_i64(dst, n, start, end, step);
  } else if constexpr (std::is_same_v<T, mllm_uint64_t>) {
    call_fill_arange_u64(dst, n, start, end, step);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_fill_arange_i16(dst, n, start, end, step);
  } else if constexpr (std::is_same_v<T, mllm_uint16_t>) {
    call_fill_arange_u16(dst, n, start, end, step);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_fill_arange_i8(dst, n, start, end, step);
  } else if constexpr (std::is_same_v<T, mllm_uint8_t>) {
    call_fill_arange_u8(dst, n, start, end, step);
  } else {
    // Fallback
    for (size_t i = 0; i < n; ++i) { dst[i] = static_cast<T>(start + i * step); }
  }
}

template<typename T>
inline void fill_random_anytype(T* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  if constexpr (std::is_same_v<T, mllm_fp32_t>) {
    call_fill_random_fp32(dst, n, start, end, seed);
  } else if constexpr (std::is_same_v<T, mllm_int32_t>) {
    call_fill_random_i32(dst, n, start, end, seed);
  } else if constexpr (std::is_same_v<T, mllm_uint32_t>) {
    call_fill_random_u32(dst, n, start, end, seed);
  } else if constexpr (std::is_same_v<T, mllm_int64_t>) {
    call_fill_random_i64(dst, n, start, end, seed);
  } else if constexpr (std::is_same_v<T, mllm_uint64_t>) {
    call_fill_random_u64(dst, n, start, end, seed);
  } else if constexpr (std::is_same_v<T, mllm_int16_t>) {
    call_fill_random_i16(dst, n, start, end, seed);
  } else if constexpr (std::is_same_v<T, mllm_uint16_t>) {
    call_fill_random_u16(dst, n, start, end, seed);
  } else if constexpr (std::is_same_v<T, mllm_int8_t>) {
    call_fill_random_i8(dst, n, start, end, seed);
  } else if constexpr (std::is_same_v<T, mllm_uint8_t>) {
    call_fill_random_u8(dst, n, start, end, seed);
  } else {
    // Fallback using LCG
    const uint64_t multiplier = 1103515245ULL;
    const uint64_t increment = 12345ULL;
    const uint64_t modulus = 1ULL << 31;
    const mllm_fp32_t range = end - start;
    uint64_t state = seed;
    for (size_t i = 0; i < n; ++i) {
      state = (multiplier * state + increment) % modulus;
      const mllm_fp32_t random_value = static_cast<mllm_fp32_t>(state) / static_cast<mllm_fp32_t>(modulus - 1);
      dst[i] = static_cast<T>(start + random_value * range);
    }
  }
}

//===----------------------------------------------------------------------===//
// Reduce
//===----------------------------------------------------------------------===//
/// Sum-reduction over a strided FP32 buffer.
/// @param dst Output buffer receiving the reduction result(s).
/// @param src Input buffer.
/// @param src_stride Stride between consecutive source elements.
/// @param size Number of elements to reduce.
/// @param thread_count Requested number of threads (implementation may clamp).
HWY_DLLEXPORT void call_reduce_sum_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t src_stride, size_t size, int32_t thread_count);

}  // namespace mllm::cpu::common

#endif
#endif  // MLLM_BACKENDS_CPU_KERNELS_COMMON_KERNEL_DISPATCH_HPP_
