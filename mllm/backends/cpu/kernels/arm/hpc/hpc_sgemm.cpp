// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/hpc/hpc_sgemm.hpp"
#include <arm_neon.h>

#include <cassert>

namespace mllm::cpu::arm {

// Optimized for decoding.
// Q: [B, H, 1, D]
// K: [B, H, S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline(const int M, const int K, const int N,
                                                            mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                                            const mllm_fp32_t* __restrict__ B,
                                                            const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                                            bool transpose_b, int thread_count) {
  assert(M == 1 && "Q must have shape [1, D]");
  const int S = N;
  const int D = K;
  assert(D % 32 == 0 && "D must be divisible by 32");

  for (int s = 0; s < S; ++s) {
    dst[s] = C[s];
    for (int d = 0; d < D; ++d) { dst[s] += A[d] * B[s * D + d]; }
  }
}

// Optimized for decoding.
// Q: [B, H, 1, D]
// K: [B, H, S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                                   const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                                   const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                                                   int thread_count) {
  assert(M == 1 && "Q (A) must have shape [1, D]");
  const int S = N;
  const int D = K;
  assert(D % 32 == 0 && "D must be divisible by 32");

  const int DTileSize = 32;
  const int DTileCount = D / DTileSize;

  const char* a_ptr = (const char*)A;
  const int prefetch_size = D * sizeof(mllm_fp32_t);
  for (int i = 0; i < prefetch_size; i += 64) { __builtin_prefetch(a_ptr + i, 0, 3); }

  for (int s = 0; s < S; ++s) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    float32x4_t acc4 = vdupq_n_f32(0.0f);
    float32x4_t acc5 = vdupq_n_f32(0.0f);
    float32x4_t acc6 = vdupq_n_f32(0.0f);
    float32x4_t acc7 = vdupq_n_f32(0.0f);

    const int s_offset = s * D;

    for (int d = 0; d < DTileCount; d += 1) {
      const int d_offset0 = d * DTileSize;

      float32x4_t a0 = vld1q_f32(A + d_offset0);
      float32x4_t b0 = vld1q_f32(B + s_offset + d_offset0);
      acc0 = vfmaq_f32(acc0, a0, b0);

      float32x4_t a1 = vld1q_f32(A + d_offset0 + 4);
      float32x4_t b1 = vld1q_f32(B + s_offset + d_offset0 + 4);
      acc1 = vfmaq_f32(acc1, a1, b1);

      float32x4_t a2 = vld1q_f32(A + d_offset0 + 8);
      float32x4_t b2 = vld1q_f32(B + s_offset + d_offset0 + 8);
      acc2 = vfmaq_f32(acc2, a2, b2);

      float32x4_t a3 = vld1q_f32(A + d_offset0 + 12);
      float32x4_t b3 = vld1q_f32(B + s_offset + d_offset0 + 12);
      acc3 = vfmaq_f32(acc3, a3, b3);

      float32x4_t a4 = vld1q_f32(A + d_offset0 + 16);
      float32x4_t b4 = vld1q_f32(B + s_offset + d_offset0 + 16);
      acc4 = vfmaq_f32(acc4, a4, b4);

      float32x4_t a5 = vld1q_f32(A + d_offset0 + 20);
      float32x4_t b5 = vld1q_f32(B + s_offset + d_offset0 + 20);
      acc5 = vfmaq_f32(acc5, a5, b5);

      float32x4_t a6 = vld1q_f32(A + d_offset0 + 24);
      float32x4_t b6 = vld1q_f32(B + s_offset + d_offset0 + 24);
      acc6 = vfmaq_f32(acc6, a6, b6);

      float32x4_t a7 = vld1q_f32(A + d_offset0 + 28);
      float32x4_t b7 = vld1q_f32(B + s_offset + d_offset0 + 28);
      acc7 = vfmaq_f32(acc7, a7, b7);
    }

    float32x4_t sum01 = vaddq_f32(acc0, acc1);
    float32x4_t sum23 = vaddq_f32(acc2, acc3);
    float32x4_t sum45 = vaddq_f32(acc4, acc5);
    float32x4_t sum67 = vaddq_f32(acc6, acc7);

    float32x4_t sum0123 = vaddq_f32(sum01, sum23);
    float32x4_t sum4567 = vaddq_f32(sum45, sum67);
    float32x4_t final_sum_vec = vaddq_f32(sum0123, sum4567);

    float result = vaddvq_f32(final_sum_vec);

    if (C) {
      dst[s] = result + C[s];
    } else {
      dst[s] = result;
    }
  }
}

}  // namespace mllm::cpu::arm