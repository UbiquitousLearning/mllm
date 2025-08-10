// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu::arm {

// Optimized for decoding.
// Q: [1, D]
// K: [S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline(const int M, const int K, const int N,
                                                            mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                                            const mllm_fp32_t* __restrict__ B,
                                                            const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                                            bool transpose_b, int thread_count);

// Optimized for decoding.
// Q: [1, D]
// K: [S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                                   const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                                   const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                                                   int thread_count);

// Optimized for decoding.
// Q: [B, H, 1, D]
// K: [B, H, S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
template<bool __enable_thread = false>
void __hpc_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk(const int BATCH, const int M, const int K, const int N,
                                                         const int Dst_batch_stride, const int A_batch_stride,
                                                         const int B_batch_stride, const int C_batch_stride,
                                                         mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                                         const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C,
                                                         bool transpose_a, bool transpose_b, int thread_count) {
  // Do not use
  // # pragma omp parallel for if (thread_count > 1)
  //                              ^^^^^^^^^^^^^^^^^^
  // Some platform(OSX) will generate inefficient code in this case. Use template instead.
  if constexpr (__enable_thread) {
#pragma omp parallel for num_threads(thread_count)
    for (int b = 0; b < BATCH; ++b) {
      auto a_ptr = A + b * A_batch_stride;
      auto b_ptr = B + b * B_batch_stride;
      auto c_ptr = C + b * C_batch_stride;
      auto d_ptr = dst + b * Dst_batch_stride;
      __hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(M, K, N, d_ptr, a_ptr, b_ptr, c_ptr, transpose_a, transpose_b, 0);
    }
  } else {
    for (int b = 0; b < BATCH; ++b) {
      auto a_ptr = A + b * A_batch_stride;
      auto b_ptr = B + b * B_batch_stride;
      auto c_ptr = C + b * C_batch_stride;
      auto d_ptr = dst + b * Dst_batch_stride;
      __hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(M, K, N, d_ptr, a_ptr, b_ptr, c_ptr, transpose_a, transpose_b, 0);
    }
  }
}

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
void __hpc_matmul_fp32_gemv_nt_nt_decode_small_d_wv_baseline(const int M, const int K, const int N,
                                                             mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                                             const mllm_fp32_t* __restrict__ B,
                                                             const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                                             bool transpose_b, int thread_count);

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
void __hpc_matmul_fp32_gemv_nt_nt_decode_small_d_wv(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                                    const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                                    const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                                                    int thread_count);

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
template<bool __enable_thread = false>
void __hpc_batch_matmul_fp32_gemv_nt_t_decode_small_d_wv(const int BATCH, const int M, const int K, const int N,
                                                         const int Dst_batch_stride, const int A_batch_stride,
                                                         const int B_batch_stride, const int C_batch_stride,
                                                         mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                                         const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C,
                                                         bool transpose_a, bool transpose_b, int thread_count) {}

void __hpc_matmul_fp32_gemv();

void __hpc_matmul_fp32_gemm();

void hpc_matmul_fp32(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                     const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                     int thread_count);

}  // namespace mllm::cpu::arm