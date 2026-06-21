// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm_blas_linear.hpp"
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_sgemm.hpp"

namespace mllm::cpu::arm {

void mllm_blas_linear_fp32(const int BATCH, const int M, const int in_channel, const int out_channel,
                           const int Dst_batch_stride, const int A_batch_stride, mllm_fp32_t* __restrict__ dst,
                           const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                           const mllm_fp32_t* __restrict__ Bias, int thread_count) {
  // Perform batch matrix multiplication:
  // dst = input @ weight.T
  mllm_blas_batch_matmul_fp32(BATCH, M, in_channel, out_channel, Dst_batch_stride, A_batch_stride, 0, 0, dst, A, B, Bias, false,
                              true, thread_count);
}

}  // namespace mllm::cpu::arm
