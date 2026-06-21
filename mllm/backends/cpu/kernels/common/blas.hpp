// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#if defined(MLLM_USE_BLAS)

// Apple Accelerate
#if defined(MLLM_BLAS_VENDOR_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace mllm::cpu::blas {

struct BLASContext {
  int n_threads = 4;
};

void matmul_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 const float* __restrict__ BIAS, int M, int N, int K, bool transpose_a, bool transpose_b);

void batch_matmul_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                       const float* __restrict__ BIAS, int Batch, int M, int N, int K, int a_batch_stride, int b_batch_stride,
                       int c_batch_stride, bool transpose_a, bool transpose_b);

}  // namespace mllm::cpu::blas

#endif
