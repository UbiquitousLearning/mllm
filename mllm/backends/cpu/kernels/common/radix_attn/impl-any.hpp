// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/kernels/common/radix_attn/arch.hpp"

#include <arm_neon.h>

namespace mllm::cpu::radix_attn::details {
template<>
struct VectorDotProduct<__AnyArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(const mllm_fp32_t* __restrict__ __lhs, const mllm_fp32_t* __restrict__ __rhs,
                                    mllm_fp32_t* __out, size_t len) {
    mllm_fp32_t ret = 0;
    for (size_t i = 0; i < len; ++i) { ret += __lhs[i] * __rhs[i]; }
    *__out = ret;
  }
};

template<>
struct MulFromConst<__AnyArchTag, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ __from, const mllm_fp32_t const_v, size_t len) {
    for (int i = 0; i < len; ++i) { __from[i] *= const_v; }
  }
};

template<>
struct FMAConstArray<__AnyArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ acc_o, const mllm_fp32_t acc_s,
                                    const mllm_fp32_t* __restrict__ v_token, size_t len) {
    for (int i = 0; i < len; ++i) { acc_o[i] += acc_s * v_token[i]; }
  }
};

template<>
struct FilledWithConst<__AnyArchTag, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ a, const mllm_fp32_t v, size_t len) {
    for (int i = 0; i < len; ++i) { a[i] = v; }
  }
};

}  // namespace mllm::cpu::radix_attn::details
