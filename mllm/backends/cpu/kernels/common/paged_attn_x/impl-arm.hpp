// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/kernels/common/paged_attn_x/arch.hpp"

#include <arm_neon.h>

namespace mllm::cpu::paged_attn_x::details {
template<>
struct VectorDotProduct<__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(const mllm_fp32_t* __restrict__ __lhs, const mllm_fp32_t* __restrict__ __rhs,
                                    mllm_fp32_t* __out, size_t len) {
    float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t main_loop_bound = len - 15;
    for (; i < main_loop_bound; i += 16) {
      const float32x4_t lhs0 = vld1q_f32(__lhs + i);
      const float32x4_t rhs0 = vld1q_f32(__rhs + i);
      sum_vec0 = vfmaq_f32(sum_vec0, lhs0, rhs0);

      const float32x4_t lhs1 = vld1q_f32(__lhs + i + 4);
      const float32x4_t rhs1 = vld1q_f32(__rhs + i + 4);
      sum_vec1 = vfmaq_f32(sum_vec1, lhs1, rhs1);

      const float32x4_t lhs2 = vld1q_f32(__lhs + i + 8);
      const float32x4_t rhs2 = vld1q_f32(__rhs + i + 8);
      sum_vec2 = vfmaq_f32(sum_vec2, lhs2, rhs2);

      const float32x4_t lhs3 = vld1q_f32(__lhs + i + 12);
      const float32x4_t rhs3 = vld1q_f32(__rhs + i + 12);
      sum_vec3 = vfmaq_f32(sum_vec3, lhs3, rhs3);
    }

    const size_t unroll4_bound = len - 3;
    for (; i < unroll4_bound; i += 4) {
      const float32x4_t lhs_vec = vld1q_f32(__lhs + i);
      const float32x4_t rhs_vec = vld1q_f32(__rhs + i);
      sum_vec0 = vfmaq_f32(sum_vec0, lhs_vec, rhs_vec);
    }

    sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
    sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
    sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);

    // Reduce
    float result = vaddvq_f32(sum_vec0);
    for (; i < len; ++i) { result += __lhs[i] * __rhs[i]; }

    *__out = result;
  }
};

template<>
struct MulFromConst<__ArmArchTag, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ __from, const mllm_fp32_t const_v, size_t len) {
    size_t i = 0;
    const size_t simd_width = 4;
    if (len >= simd_width) {
      float32x4_t const_vec = vdupq_n_f32(const_v);
      size_t simd_len = (len / simd_width) * simd_width;
      for (; i < simd_len; i += simd_width) {
        float32x4_t data_vec = vld1q_f32(&__from[i]);
        data_vec = vmulq_f32(data_vec, const_vec);
        vst1q_f32(&__from[i], data_vec);
      }
    }
    for (; i < len; ++i) { __from[i] *= const_v; }
  }
};

template<>
struct FMAConstArray<__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ acc_o, const mllm_fp32_t acc_s,
                                    const mllm_fp32_t* __restrict__ v_token, size_t len) {
    size_t i = 0;
    const size_t simd_width = 4;

    if (len >= simd_width) {
      float32x4_t acc_s_vec = vdupq_n_f32(acc_s);

      size_t simd_len = (len / simd_width) * simd_width;
      for (; i < simd_len; i += simd_width) {
        float32x4_t v_token_vec = vld1q_f32(&v_token[i]);
        float32x4_t acc_o_vec = vld1q_f32(&acc_o[i]);
        acc_o_vec = vfmaq_f32(acc_o_vec, acc_s_vec, v_token_vec);
        vst1q_f32(&acc_o[i], acc_o_vec);
      }
    }
    for (; i < len; ++i) { acc_o[i] += acc_s * v_token[i]; }
  }
};

}  // namespace mllm::cpu::paged_attn_x::details
