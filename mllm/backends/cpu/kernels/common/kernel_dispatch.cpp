// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/utils/CPUArchHelper.hpp"
#if !(defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM))

#include "mllm/backends/cpu/kernels/common/kernel_dispatch.hpp"

// >>>> for dynamic dispatch only, skip if you want static dispatch
// First undef to prevent error when re-included.
#undef HWY_TARGET_INCLUDE
// For dynamic dispatch, specify the name of the current file (unfortunately
// __FILE__ is not reliable) so that foreach_target.h can re-include it.
#define HWY_TARGET_INCLUDE "mllm/backends/cpu/kernels/common/kernel_dispatch.cpp"
// Generates code for each enabled target by re-including this source file.
#include <hwy/foreach_target.h>  // IWYU pragma: keep
// <<<< end of dynamic dispatch

// Include all inline implementations here
#include "mllm/backends/cpu/kernels/common/elewise-inl.hpp"
#include "mllm/backends/cpu/kernels/common/fill-inl.hpp"

#if HWY_ONCE
namespace mllm::cpu::common {

//===----------------------------------------------------------------------===//
// Element-wise
//===----------------------------------------------------------------------===//
HWY_EXPORT(elewise_add_fp32);
HWY_EXPORT(elewise_sub_fp32);
HWY_EXPORT(elewise_mul_fp32);
HWY_EXPORT(elewise_div_fp32);
HWY_EXPORT(elewise_add_scalar_fp32);
HWY_EXPORT(elewise_sub_scalar_fp32);
HWY_EXPORT(elewise_mul_scalar_fp32);
HWY_EXPORT(elewise_div_scalar_fp32);

HWY_DLLEXPORT void call_elewise_add_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  HWY_DYNAMIC_DISPATCH(elewise_add_fp32)(out, x, y, n);
}

HWY_DLLEXPORT void call_elewise_sub_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  HWY_DYNAMIC_DISPATCH(elewise_sub_fp32)(out, x, y, n);
}

HWY_DLLEXPORT void call_elewise_mul_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  HWY_DYNAMIC_DISPATCH(elewise_mul_fp32)(out, x, y, n);
}

HWY_DLLEXPORT void call_elewise_div_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  HWY_DYNAMIC_DISPATCH(elewise_div_fp32)(out, x, y, n);
}

HWY_DLLEXPORT void call_elewise_add_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n) {
  HWY_DYNAMIC_DISPATCH(elewise_add_scalar_fp32)(out, x, y, n);
}

HWY_DLLEXPORT void call_elewise_sub_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n) {
  HWY_DYNAMIC_DISPATCH(elewise_sub_scalar_fp32)(out, x, y, n);
}

HWY_DLLEXPORT void call_elewise_mul_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n) {
  HWY_DYNAMIC_DISPATCH(elewise_mul_scalar_fp32)(out, x, y, n);
}

HWY_DLLEXPORT void call_elewise_div_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n) {
  HWY_DYNAMIC_DISPATCH(elewise_div_scalar_fp32)(out, x, y, n);
}

//===----------------------------------------------------------------------===//
// GELU
//===----------------------------------------------------------------------===//
// HWY_EXPORT(gelu_fp32);
//
// HWY_DLLEXPORT void call_gelu_fp32(mllm_fp32_t* out, const mllm_fp32_t* in, size_t n) {
//   HWY_DYNAMIC_DISPATCH(gelu_fp32)(out, in, n);
// }

//===----------------------------------------------------------------------===//
// Fill Zeros
//===----------------------------------------------------------------------===//
HWY_EXPORT(fill_zeros_fp32);
HWY_EXPORT(fill_zeros_fp64);
HWY_EXPORT(fill_zeros_i32);
HWY_EXPORT(fill_zeros_u32);
HWY_EXPORT(fill_zeros_i64);
HWY_EXPORT(fill_zeros_u64);
HWY_EXPORT(fill_zeros_i16);
HWY_EXPORT(fill_zeros_u16);
HWY_EXPORT(fill_zeros_i8);
HWY_EXPORT(fill_zeros_u8);

HWY_DLLEXPORT void call_fill_zeros_fp32(mllm_fp32_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_fp32)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_fp64(mllm_fp64_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_fp64)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_i32(mllm_int32_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_i32)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_u32(mllm_uint32_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_u32)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_i64(mllm_int64_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_i64)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_u64(mllm_uint64_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_u64)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_i16(mllm_int16_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_i16)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_u16(mllm_uint16_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_u16)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_i8(mllm_int8_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_i8)(dst, n); }
HWY_DLLEXPORT void call_fill_zeros_u8(mllm_uint8_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_zeros_u8)(dst, n); }

//===----------------------------------------------------------------------===//
// Fill Ones
//===----------------------------------------------------------------------===//
HWY_EXPORT(fill_ones_fp32);
HWY_EXPORT(fill_ones_fp64);
HWY_EXPORT(fill_ones_i32);
HWY_EXPORT(fill_ones_u32);
HWY_EXPORT(fill_ones_i64);
HWY_EXPORT(fill_ones_u64);
HWY_EXPORT(fill_ones_i16);
HWY_EXPORT(fill_ones_u16);
HWY_EXPORT(fill_ones_i8);
HWY_EXPORT(fill_ones_u8);

HWY_DLLEXPORT void call_fill_ones_fp32(mllm_fp32_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_fp32)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_fp64(mllm_fp64_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_fp64)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_i32(mllm_int32_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_i32)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_u32(mllm_uint32_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_u32)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_i64(mllm_int64_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_i64)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_u64(mllm_uint64_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_u64)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_i16(mllm_int16_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_i16)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_u16(mllm_uint16_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_u16)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_i8(mllm_int8_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_i8)(dst, n); }
HWY_DLLEXPORT void call_fill_ones_u8(mllm_uint8_t* dst, size_t n) { HWY_DYNAMIC_DISPATCH(fill_ones_u8)(dst, n); }

//===----------------------------------------------------------------------===//
// Fill Specific Value
//===----------------------------------------------------------------------===//
HWY_EXPORT(fill_value_fp32);
HWY_EXPORT(fill_value_fp64);
HWY_EXPORT(fill_value_i32);
HWY_EXPORT(fill_value_u32);
HWY_EXPORT(fill_value_i64);
HWY_EXPORT(fill_value_u64);
HWY_EXPORT(fill_value_i16);
HWY_EXPORT(fill_value_u16);
HWY_EXPORT(fill_value_i8);
HWY_EXPORT(fill_value_u8);

HWY_DLLEXPORT void call_fill_value_fp32(mllm_fp32_t* dst, size_t n, mllm_fp32_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_fp32)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_fp64(mllm_fp64_t* dst, size_t n, mllm_fp64_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_fp64)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_i32(mllm_int32_t* dst, size_t n, mllm_int32_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_i32)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_u32(mllm_uint32_t* dst, size_t n, mllm_uint32_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_u32)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_i64(mllm_int64_t* dst, size_t n, mllm_int64_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_i64)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_u64(mllm_uint64_t* dst, size_t n, mllm_uint64_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_u64)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_i16(mllm_int16_t* dst, size_t n, mllm_int16_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_i16)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_u16(mllm_uint16_t* dst, size_t n, mllm_uint16_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_u16)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_i8(mllm_int8_t* dst, size_t n, mllm_int8_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_i8)(dst, n, value);
}
HWY_DLLEXPORT void call_fill_value_u8(mllm_uint8_t* dst, size_t n, mllm_uint8_t value) {
  HWY_DYNAMIC_DISPATCH(fill_value_u8)(dst, n, value);
}

//===----------------------------------------------------------------------===//
// Fill Arange
//===----------------------------------------------------------------------===//
HWY_EXPORT(fill_arange_fp32);
HWY_EXPORT(fill_arange_i32);
HWY_EXPORT(fill_arange_u32);
HWY_EXPORT(fill_arange_i64);
HWY_EXPORT(fill_arange_u64);
HWY_EXPORT(fill_arange_i16);
HWY_EXPORT(fill_arange_u16);
HWY_EXPORT(fill_arange_i8);
HWY_EXPORT(fill_arange_u8);

HWY_DLLEXPORT void call_fill_arange_fp32(mllm_fp32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_fp32)(dst, n, start, end, step);
}
HWY_DLLEXPORT void call_fill_arange_i32(mllm_int32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_i32)(dst, n, start, end, step);
}
HWY_DLLEXPORT void call_fill_arange_u32(mllm_uint32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_u32)(dst, n, start, end, step);
}
HWY_DLLEXPORT void call_fill_arange_i64(mllm_int64_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_i64)(dst, n, start, end, step);
}
HWY_DLLEXPORT void call_fill_arange_u64(mllm_uint64_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_u64)(dst, n, start, end, step);
}
HWY_DLLEXPORT void call_fill_arange_i16(mllm_int16_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_i16)(dst, n, start, end, step);
}
HWY_DLLEXPORT void call_fill_arange_u16(mllm_uint16_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_u16)(dst, n, start, end, step);
}
HWY_DLLEXPORT void call_fill_arange_i8(mllm_int8_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_i8)(dst, n, start, end, step);
}
HWY_DLLEXPORT void call_fill_arange_u8(mllm_uint8_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, mllm_fp32_t step) {
  HWY_DYNAMIC_DISPATCH(fill_arange_u8)(dst, n, start, end, step);
}

//===----------------------------------------------------------------------===//
// Fill Random
//===----------------------------------------------------------------------===//
HWY_EXPORT(fill_random_fp32);
HWY_EXPORT(fill_random_i32);
HWY_EXPORT(fill_random_u32);
HWY_EXPORT(fill_random_i64);
HWY_EXPORT(fill_random_u64);
HWY_EXPORT(fill_random_i16);
HWY_EXPORT(fill_random_u16);
HWY_EXPORT(fill_random_i8);
HWY_EXPORT(fill_random_u8);

HWY_DLLEXPORT void call_fill_random_fp32(mllm_fp32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_fp32)(dst, n, start, end, seed);
}
HWY_DLLEXPORT void call_fill_random_i32(mllm_int32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_i32)(dst, n, start, end, seed);
}
HWY_DLLEXPORT void call_fill_random_u32(mllm_uint32_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_u32)(dst, n, start, end, seed);
}
HWY_DLLEXPORT void call_fill_random_i64(mllm_int64_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_i64)(dst, n, start, end, seed);
}
HWY_DLLEXPORT void call_fill_random_u64(mllm_uint64_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_u64)(dst, n, start, end, seed);
}
HWY_DLLEXPORT void call_fill_random_i16(mllm_int16_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_i16)(dst, n, start, end, seed);
}
HWY_DLLEXPORT void call_fill_random_u16(mllm_uint16_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_u16)(dst, n, start, end, seed);
}
HWY_DLLEXPORT void call_fill_random_i8(mllm_int8_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_i8)(dst, n, start, end, seed);
}
HWY_DLLEXPORT void call_fill_random_u8(mllm_uint8_t* dst, size_t n, mllm_fp32_t start, mllm_fp32_t end, uint64_t seed) {
  HWY_DYNAMIC_DISPATCH(fill_random_u8)(dst, n, start, end, seed);
}

}  // namespace mllm::cpu::common

#endif  // HWY_ONCE
#endif
