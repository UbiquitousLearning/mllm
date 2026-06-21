// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/blas.hpp"

#if defined(MLLM_USE_BLAS)

namespace mllm::cpu::blas {

void matmul_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 const float* __restrict__ BIAS, int M, int N, int K, bool transpose_a, bool transpose_b) {
#if defined(MLLM_BLAS_VENDOR_ACCELERATE)
  const enum CBLAS_TRANSPOSE TransA = transpose_a ? CblasTrans : CblasNoTrans;
  const enum CBLAS_TRANSPOSE TransB = transpose_b ? CblasTrans : CblasNoTrans;

  const int lda = transpose_a ? M : K;
  const int ldb = transpose_b ? K : N;

  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, N);

  if (BIAS != nullptr) {
    for (int i = 0; i < M; ++i) {
      float* c_row = C + i * N;
      // cblas_saxpy will do y = a * x + y
      cblas_saxpy(N, 1.0f, BIAS, 1, c_row, 1);
    }
  }
#else
  NYI("mllm::cpu::blas::matmul_fp32 is not ENABLED. Set MLLM_USE_BLAS=ON and vendors[MKL, Accelerate, BLIS] to enable it");
#endif
}

void batch_matmul_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                       const float* __restrict__ BIAS, int Batch, int M, int N, int K, int a_batch_stride, int b_batch_stride,
                       int c_batch_stride, bool transpose_a, bool transpose_b) {
  for (int i = 0; i < Batch; ++i) {
    const float* current_A = A + i * a_batch_stride;
    const float* current_B = B + i * b_batch_stride;
    float* current_C = C + i * c_batch_stride;
    matmul_fp32(current_A, current_B, current_C, BIAS, M, N, K, transpose_a, transpose_b);
  }
}

}  // namespace mllm::cpu::blas

#endif
