// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <hwy/highway.h>

#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1/arch.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1/impl-any-simd-inst.hpp"

namespace mllm::cpu::flash_attn2::details {

template<>
struct VectorDotProduct<__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(const mllm_fp32_t* __restrict__ __lhs, const mllm_fp32_t* __restrict__ __rhs,
                                    mllm_fp32_t* __out, size_t len) {
    call_vector_dot_product_fp32_fp32_fp32(__lhs, __rhs, __out, len);
  }
};

template<>
struct MulFromConst<__X86ArchTag, mllm_fp32_t, mllm_fp32_t> {
  static void run(mllm_fp32_t* from, mllm_fp32_t c, size_t len) { call_mul_from_const_fp32(from, c, len); }
};

template<>
struct FMAConstArray<__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t> {
  static void run(mllm_fp32_t* acc_o, mllm_fp32_t acc_s, const mllm_fp32_t* v_token, size_t len) {
    call_fma_const_array_fp32(acc_o, acc_s, v_token, len);
  }
};

template<>
struct FilledWithConst<__X86ArchTag, mllm_fp32_t> {
  static void run(mllm_fp32_t* a, mllm_fp32_t v, size_t len) { call_filled_with_const_fp32(a, v, len); }
};

}  // namespace mllm::cpu::flash_attn2::details
