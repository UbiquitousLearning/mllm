// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#ifndef MLLM_BACKENDS_CPU_KERNELS_COMMON_KERNEL_DISPATCH_HPP_
#define MLLM_BACKENDS_CPU_KERNELS_COMMON_KERNEL_DISPATCH_HPP_

#include "mllm/utils/CPUArchHelper.hpp"
#if !(defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM))

#include "mllm/core/DataTypes.hpp"

// Platform-specific definitions used for declaring an interface, independent of
// the SIMD instruction set.
#include <hwy/base.h>  // HWY_RESTRICT
namespace mllm::cpu::common {

//===----------------------------------------------------------------------===//
// Elementwise + - * / By Matrix
//===----------------------------------------------------------------------===//
HWY_DLLEXPORT void call_elewise_add_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n);
HWY_DLLEXPORT void call_elewise_div_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n);

//===----------------------------------------------------------------------===//
// Elementwise + - * / By Const
//===----------------------------------------------------------------------===//
HWY_DLLEXPORT void call_elewise_add_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_sub_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_mul_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n);
HWY_DLLEXPORT void call_elewise_div_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, mllm_fp32_t y, size_t n);

}  // namespace mllm::cpu::common

#endif
#endif  // MLLM_BACKENDS_CPU_KERNELS_COMMON_KERNEL_DISPATCH_HPP_
