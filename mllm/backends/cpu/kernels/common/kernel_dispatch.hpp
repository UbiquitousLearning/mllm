// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <hwy/base.h>  // HWY_DLLEXPORT
#include "mllm/core/DataTypes.hpp"

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
