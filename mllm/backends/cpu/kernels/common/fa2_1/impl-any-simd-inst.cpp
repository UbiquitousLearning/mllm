// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/utils/CPUArchHelper.hpp"
#if !(defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM))

#include "mllm/backends/cpu/kernels/common/fa2_1/impl-any-simd-inst.hpp"

// >>>> for dynamic dispatch only, skip if you want static dispatch
// First undef to prevent error when re-included.
#undef HWY_TARGET_INCLUDE
// For dynamic dispatch, specify the name of the current file (unfortunately
// __FILE__ is not reliable) so that foreach_target.h can re-include it.
#define HWY_TARGET_INCLUDE "mllm/backends/cpu/kernels/common/fa2_1/impl-any-simd-inst.cpp"
// Generates code for each enabled target by re-including this source file.
#include <hwy/foreach_target.h>  // IWYU pragma: keep
// <<<< end of dynamic dispatch

#include "mllm/backends/cpu/kernels/common/fa2_1/impl-any-simd-inl.hpp"

// The table of pointers to the various implementations in HWY_NAMESPACE must
// be compiled only once (foreach_target #includes this file multiple times).
// HWY_ONCE is true for only one of these 'compilation passes'.
#if HWY_ONCE

namespace mllm::cpu::flash_attn2::details {

HWY_EXPORT(vector_dot_product_fp32_fp32_fp32);
HWY_EXPORT(mul_from_const_fp32);
HWY_EXPORT(fma_const_array_fp32);
HWY_EXPORT(filled_with_const_fp32);

HWY_DLLEXPORT void call_vector_dot_product_fp32_fp32_fp32(const mllm_fp32_t* HWY_RESTRICT lhs,
                                                          const mllm_fp32_t* HWY_RESTRICT rhs, mllm_fp32_t* HWY_RESTRICT out,
                                                          size_t len) {
  HWY_DYNAMIC_DISPATCH(vector_dot_product_fp32_fp32_fp32)(lhs, rhs, out, len);
}

HWY_DLLEXPORT void call_fma_const_array_fp32(mllm_fp32_t* HWY_RESTRICT acc_o, mllm_fp32_t acc_s,
                                             const mllm_fp32_t* HWY_RESTRICT v_token, size_t len) {
  HWY_DYNAMIC_DISPATCH(fma_const_array_fp32)(acc_o, acc_s, v_token, len);
}

HWY_DLLEXPORT void call_filled_with_const_fp32(mllm_fp32_t* HWY_RESTRICT a, float v, size_t len) {
  HWY_DYNAMIC_DISPATCH(filled_with_const_fp32)(a, v, len);
}

HWY_DLLEXPORT void call_mul_from_const_fp32(mllm_fp32_t* HWY_RESTRICT from, float c, size_t len) {
  HWY_DYNAMIC_DISPATCH(mul_from_const_fp32)(from, c, len);
}

}  // namespace mllm::cpu::flash_attn2::details
#endif  // HWY_ONCE
#endif
