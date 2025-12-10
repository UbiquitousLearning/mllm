// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/kernel_dispatch.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

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


}  // namespace mllm::cpu::common

#endif  // HWY_ONCE
