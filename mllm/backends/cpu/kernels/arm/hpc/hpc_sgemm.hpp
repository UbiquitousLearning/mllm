// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/DataTypes.hpp"

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
                                                            bool transpose_b, int thread_count);

// Optimized for decoding.
// Q: [B, H, 1, D]
// K: [B, H, S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                                   const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                                   const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                                                   int thread_count);

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
void __hpc_matmul_fp32_gemv_nt_nt_decode_small_d_wv();

void __hpc_matmul_fp32_gemv();

void __hpc_matmul_fp32_gemm();

void hpc_matmul_fp32(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                     const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                     int thread_count);

}  // namespace mllm::cpu::arm