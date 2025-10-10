// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/kernels/common/radix_attn/arch.hpp"

#include <arm_neon.h>

namespace mllm::cpu::radix_attn::details {
template<>
struct VectorDotProduct<__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(const mllm_fp32_t* __restrict__ __lhs, const mllm_fp32_t* __restrict__ __rhs,
                                    mllm_fp32_t* __out, size_t len) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t block_size = 16;
    const size_t len_aligned = len & ~(block_size - 1);

    for (; i < len_aligned; i += block_size) {
      float32x4_t lhs_vec0 = vld1q_f32(__lhs + i);
      float32x4_t lhs_vec1 = vld1q_f32(__lhs + i + 4);
      float32x4_t lhs_vec2 = vld1q_f32(__lhs + i + 8);
      float32x4_t lhs_vec3 = vld1q_f32(__lhs + i + 12);

      float32x4_t rhs_vec0 = vld1q_f32(__rhs + i);
      float32x4_t rhs_vec1 = vld1q_f32(__rhs + i + 4);
      float32x4_t rhs_vec2 = vld1q_f32(__rhs + i + 8);
      float32x4_t rhs_vec3 = vld1q_f32(__rhs + i + 12);

      sum_vec = vfmaq_f32(sum_vec, lhs_vec0, rhs_vec0);
      sum_vec = vfmaq_f32(sum_vec, lhs_vec1, rhs_vec1);
      sum_vec = vfmaq_f32(sum_vec, lhs_vec2, rhs_vec2);
      sum_vec = vfmaq_f32(sum_vec, lhs_vec3, rhs_vec3);
    }

    for (; i + 3 < len; i += 4) {
      float32x4_t lhs_vec = vld1q_f32(__lhs + i);
      float32x4_t rhs_vec = vld1q_f32(__rhs + i);
      sum_vec = vfmaq_f32(sum_vec, lhs_vec, rhs_vec);
    }

    float result = vaddvq_f32(sum_vec);

    for (; i < len; ++i) { result += __lhs[i] * __rhs[i]; }

    *__out = result;
  }
};

template<>
struct MulFromConst<__ArmArchTag, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ __from, const mllm_fp32_t const_v, size_t len) {
    float32x4_t const_vec = vdupq_n_f32(const_v);

    size_t i = 0;
    const size_t block_size = 16;
    const size_t len_aligned = len & ~(block_size - 1);

    for (; i < len_aligned; i += block_size) {
      float32x4_t vec0 = vld1q_f32(__from + i);
      float32x4_t vec1 = vld1q_f32(__from + i + 4);
      float32x4_t vec2 = vld1q_f32(__from + i + 8);
      float32x4_t vec3 = vld1q_f32(__from + i + 12);

      // FIXME: FMA may be muster than MUL
      vec0 = vmulq_f32(vec0, const_vec);
      vec1 = vmulq_f32(vec1, const_vec);
      vec2 = vmulq_f32(vec2, const_vec);
      vec3 = vmulq_f32(vec3, const_vec);

      vst1q_f32(__from + i, vec0);
      vst1q_f32(__from + i + 4, vec1);
      vst1q_f32(__from + i + 8, vec2);
      vst1q_f32(__from + i + 12, vec3);
    }

    for (; i + 3 < len; i += 4) {
      float32x4_t vec = vld1q_f32(__from + i);
      vec = vmulq_f32(vec, const_vec);
      vst1q_f32(__from + i, vec);
    }

    for (; i < len; ++i) { __from[i] *= const_v; }
  }
};

template<>
struct FMAConstArray<__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ acc_o, const mllm_fp32_t acc_s,
                                    const mllm_fp32_t* __restrict__ v_token, size_t len) {
    float32x4_t acc_vec = vdupq_n_f32(acc_s);

    size_t i = 0;
    const size_t block_size = 16;
    const size_t len_aligned = len & ~(block_size - 1);

    for (; i < len_aligned; i += block_size) {
      float32x4_t acc0 = vld1q_f32(acc_o + i);
      float32x4_t token0 = vld1q_f32(v_token + i);
      acc0 = vfmaq_f32(acc0, token0, acc_vec);
      vst1q_f32(acc_o + i, acc0);

      float32x4_t acc1 = vld1q_f32(acc_o + i + 4);
      float32x4_t token1 = vld1q_f32(v_token + i + 4);
      acc1 = vfmaq_f32(acc1, token1, acc_vec);
      vst1q_f32(acc_o + i + 4, acc1);

      float32x4_t acc2 = vld1q_f32(acc_o + i + 8);
      float32x4_t token2 = vld1q_f32(v_token + i + 8);
      acc2 = vfmaq_f32(acc2, token2, acc_vec);
      vst1q_f32(acc_o + i + 8, acc2);

      float32x4_t acc3 = vld1q_f32(acc_o + i + 12);
      float32x4_t token3 = vld1q_f32(v_token + i + 12);
      acc3 = vfmaq_f32(acc3, token3, acc_vec);
      vst1q_f32(acc_o + i + 12, acc3);
    }

    for (; i + 3 < len; i += 4) {
      float32x4_t acc = vld1q_f32(acc_o + i);
      float32x4_t token = vld1q_f32(v_token + i);
      acc = vfmaq_f32(acc, token, acc_vec);
      vst1q_f32(acc_o + i, acc);
    }

    for (; i < len; ++i) { acc_o[i] += acc_s * v_token[i]; }
  }
};

template<>
struct FilledWithConst<__ArmArchTag, mllm_fp32_t> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ a, const mllm_fp32_t v, size_t len) {
    float32x4_t const_vec = vdupq_n_f32(v);

    size_t i = 0;
    const size_t block_size = 16;
    const size_t len_aligned = len & ~(block_size - 1);

    for (; i < len_aligned; i += block_size) {
      vst1q_f32(a + i, const_vec);
      vst1q_f32(a + i + 4, const_vec);
      vst1q_f32(a + i + 8, const_vec);
      vst1q_f32(a + i + 12, const_vec);
    }

    for (; i + 3 < len; i += 4) { vst1q_f32(a + i, const_vec); }

    for (; i < len; ++i) { a[i] = v; }
  }
};

}  // namespace mllm::cpu::radix_attn::details
