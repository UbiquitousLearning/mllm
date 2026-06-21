// Copyright (c) MLLM Team.
// Licensed under the MIT License.
// Normal header with include guard and namespace.
#ifndef HIGHWAY_HWY_FA2_IMPL_INL_H_
#define HIGHWAY_HWY_FA2_IMPL_INL_H_

#include "mllm/utils/CPUArchHelper.hpp"
#if !(defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM))

#include "mllm/core/DataTypes.hpp"

// Platform-specific definitions used for declaring an interface, independent of
// the SIMD instruction set.
#include <hwy/base.h>  // HWY_RESTRICT

namespace mllm::cpu::flash_attn2::details {

HWY_DLLEXPORT void call_vector_dot_product_fp32_fp32_fp32(const mllm_fp32_t* HWY_RESTRICT lhs,
                                                          const mllm_fp32_t* HWY_RESTRICT rhs, mllm_fp32_t* HWY_RESTRICT out,
                                                          size_t len);

HWY_DLLEXPORT void call_fma_const_array_fp32(mllm_fp32_t* HWY_RESTRICT acc_o, mllm_fp32_t acc_s,
                                             const mllm_fp32_t* HWY_RESTRICT v_token, size_t len);

HWY_DLLEXPORT void call_filled_with_const_fp32(mllm_fp32_t* HWY_RESTRICT a, float v, size_t len);

HWY_DLLEXPORT void call_mul_from_const_fp32(mllm_fp32_t* HWY_RESTRICT from, float c, size_t len);

}  // namespace mllm::cpu::flash_attn2::details

#endif  // HIGHWAY_HWY_FA2_IMPL_INL_H_
#endif
