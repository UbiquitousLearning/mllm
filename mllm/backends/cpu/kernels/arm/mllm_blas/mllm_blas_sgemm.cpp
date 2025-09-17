// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <arm_neon.h>
#include <algorithm>
#include <cassert>

#include "mllm/utils/Common.hpp"
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_sgemm.hpp"

namespace mllm::cpu::arm {

// Optimized for decoding.
// Q: [1, D]
// K: [S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline(const int M, const int K, const int N,
                                                                  mllm_fp32_t* __restrict__ dst,
                                                                  const mllm_fp32_t* __restrict__ A,
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
// Q: [1, D]
// K: [S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
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

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv_baseline(const int M, const int K, const int N,
                                                                   mllm_fp32_t* __restrict__ dst,
                                                                   const mllm_fp32_t* __restrict__ A,
                                                                   const mllm_fp32_t* __restrict__ B,
                                                                   const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                                                   bool transpose_b, int thread_count) {
  for (int n = 0; n < N; ++n) {
    float sum = 0.0f;
    if (C != nullptr) { sum = C[n]; }
    for (int k = 0; k < K; ++k) { sum += A[k] * B[k * N + n]; }
    dst[n] = sum;
  }
}

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                                          const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                                          const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                                                          int thread_count) {
  if (C != nullptr) {
    for (int n = 0; n < N; ++n) { dst[n] = C[n]; }
  } else {
    for (int n = 0; n < N; ++n) { dst[n] = 0.0f; }
  }

  int k = 0;
  for (; k <= K - 4; k += 4) {
    float32x4_t a_vec = vld1q_f32(A + k);
    int n = 0;
    for (; n <= N - 4; n += 4) {
      float32x4_t b_vec0 = vld1q_f32(B + (k + 0) * N + n);
      float32x4_t b_vec1 = vld1q_f32(B + (k + 1) * N + n);
      float32x4_t b_vec2 = vld1q_f32(B + (k + 2) * N + n);
      float32x4_t b_vec3 = vld1q_f32(B + (k + 3) * N + n);

      float32x4_t dst_vec = vld1q_f32(dst + n);

      dst_vec = vmlaq_laneq_f32(dst_vec, b_vec0, a_vec, 0);
      dst_vec = vmlaq_laneq_f32(dst_vec, b_vec1, a_vec, 1);
      dst_vec = vmlaq_laneq_f32(dst_vec, b_vec2, a_vec, 2);
      dst_vec = vmlaq_laneq_f32(dst_vec, b_vec3, a_vec, 3);

      vst1q_f32(dst + n, dst_vec);
    }

    for (; n < N; ++n) {
      float sum = dst[n];
      sum += A[k + 0] * B[(k + 0) * N + n];
      sum += A[k + 1] * B[(k + 1) * N + n];
      sum += A[k + 2] * B[(k + 2) * N + n];
      sum += A[k + 3] * B[(k + 3) * N + n];
      dst[n] = sum;
    }
  }

  for (; k < K; ++k) {
    float a_val = A[k];
    int n = 0;
    for (; n <= N - 4; n += 4) {
      float32x4_t b_vec = vld1q_f32(B + k * N + n);
      float32x4_t dst_vec = vld1q_f32(dst + n);
      dst_vec = vmlaq_n_f32(dst_vec, b_vec, a_val);
      vst1q_f32(dst + n, dst_vec);
    }
    for (; n < N; ++n) { dst[n] += a_val * B[k * N + n]; }
  }
}

void __mllm_blas_matmul_fp32_gemv(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                  const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                  const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b, int thread_count) {
  if (!transpose_a && transpose_b) {
    __mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk(M, K, N, dst, A, B, C, transpose_a, transpose_b, thread_count);
  } else if (!transpose_a && !transpose_b) {
    __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv(M, K, N, dst, A, B, C, transpose_a, transpose_b, thread_count);
  } else {
    NYI("transpose_a && transpose_b");
  }
}

void __mllm_blas_batch_matmul_fp32_gemv(const int BATCH, const int M, const int K, const int N, const int Dst_batch_stride,
                                        const int A_batch_stride, const int B_batch_stride, const int C_batch_stride,
                                        mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                        const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                        bool transpose_b, int thread_count) {
  if (!transpose_a && transpose_b) {
    if (thread_count > 1) {
      __mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk<true>(BATCH, M, K, N, Dst_batch_stride, A_batch_stride,
                                                                      B_batch_stride, C_batch_stride, dst, A, B, C, transpose_a,
                                                                      transpose_b, thread_count);

    } else {
      __mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk<false>(BATCH, M, K, N, Dst_batch_stride, A_batch_stride,
                                                                       B_batch_stride, C_batch_stride, dst, A, B, C,
                                                                       transpose_a, transpose_b, thread_count);
    }
  } else if (!transpose_a && !transpose_b) {
    if (thread_count > 1) {
      __mllm_blas_batch_matmul_fp32_gemv_nt_nt_decode_small_d_wv<true>(BATCH, M, K, N, Dst_batch_stride, A_batch_stride,
                                                                       B_batch_stride, C_batch_stride, dst, A, B, C,
                                                                       transpose_a, transpose_b, thread_count);
    } else {
      __mllm_blas_batch_matmul_fp32_gemv_nt_nt_decode_small_d_wv<false>(BATCH, M, K, N, Dst_batch_stride, A_batch_stride,
                                                                        B_batch_stride, C_batch_stride, dst, A, B, C,
                                                                        transpose_a, transpose_b, thread_count);
    }
  } else {
    NYI("transpose_a && transpose_b");
  }
}

namespace {
__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
static inline void dispatch_tile(int rm, int rn, const float* a, int64_t lda, const float* b, int64_t ldb, float* c,
                                 int64_t ldc, int64_t k) {
#define KERNEL(__tile_m, __tile_n) \
  case (__tile_m << 8) | __tile_n: MicroKernel<__tile_m, __tile_n>::accumulate(a, lda, b, ldb, c, ldc, k); break;

  switch ((std::min(rm, 8) << 8) | std::min(rn, 16)) {
    // Instanced
    KERNEL(8, 16)
    KERNEL(4, 16)
    KERNEL(1, 4)
    // General GEMV, M = 1, decode
    KERNEL(1, 1)
    KERNEL(1, 2)
    KERNEL(1, 3)
    KERNEL(1, 5)
    KERNEL(1, 6)
    KERNEL(1, 7)
    KERNEL(1, 8)
    KERNEL(1, 9)
    KERNEL(1, 10)
    KERNEL(1, 11)
    KERNEL(1, 12)
    KERNEL(1, 13)
    KERNEL(1, 14)
    KERNEL(1, 15)
    KERNEL(1, 16)
    // Compiler Optimized Kernel
    KERNEL(2, 2)
    KERNEL(2, 4)
    KERNEL(2, 6)
    KERNEL(2, 8)
    KERNEL(2, 12)
    KERNEL(2, 16)
    KERNEL(4, 2)
    KERNEL(4, 4)
    KERNEL(4, 6)
    KERNEL(4, 8)
    KERNEL(4, 12)
    default: {
      auto _rm = std::min(rm, 8);
      auto _rn = std::min(rn, 16);
      for (int i = 0; i < _rm; ++i) {
        for (int j = 0; j < _rn; ++j) { c[i * ldc + j] = 0; }
      }
      for (int64_t l = 0; l < k; ++l) {
        for (int i = 0; i < _rm; ++i) {
          const float ai = a[i * lda + l];
          for (int j = 0; j < _rn; ++j) { c[i * ldc + j] += ai * b[l * ldb + j]; }
        }
      }
      break;
    }
  }

#undef KERNEL
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
static inline void dispatch_tile_nt_t(int rm, int rn, const float* a, int64_t lda, const float* b, int64_t ldb, float* c,
                                      int64_t ldc, int64_t k, const float* bias) {
#define KERNEL(__tile_m, __tile_n)                                                          \
  case (__tile_m << 8) | __tile_n:                                                          \
    MicroKernel_NT_T_Bias<__tile_m, __tile_n>::accumulate(a, lda, b, ldb, c, ldc, k, bias); \
    break;

  switch ((std::min(rm, 8) << 8) | std::min(rn, 16)) {
    // Compiler Optimized Kernel
    KERNEL(8, 16)
    KERNEL(4, 16)
    KERNEL(1, 4)
    // General GEMV, M = 1, decode
    KERNEL(1, 1)
    KERNEL(1, 2)
    KERNEL(1, 3)
    KERNEL(1, 5)
    KERNEL(1, 6)
    KERNEL(1, 7)
    KERNEL(1, 8)
    KERNEL(1, 9)
    KERNEL(1, 10)
    KERNEL(1, 11)
    KERNEL(1, 12)
    KERNEL(1, 13)
    KERNEL(1, 14)
    KERNEL(1, 15)
    KERNEL(1, 16)
    // Compiler Optimized Kernel
    KERNEL(2, 2)
    KERNEL(2, 4)
    KERNEL(2, 6)
    KERNEL(2, 8)
    KERNEL(2, 12)
    KERNEL(2, 16)
    KERNEL(4, 2)
    KERNEL(4, 4)
    KERNEL(4, 6)
    KERNEL(4, 8)
    KERNEL(4, 12)
    default: {
      auto _rm = std::min(rm, 8);
      auto _rn = std::min(rn, 16);
      for (int i = 0; i < _rm; ++i) {
        for (int j = 0; j < _rn; ++j) {
          float sum = 0.0f;
          for (int64_t l = 0; l < k; ++l) { sum += a[i * lda + l] * b[j * ldb + l]; }
          c[i * ldc + j] = sum;
        }
      }
      if (bias != nullptr) {
        for (int i = 0; i < _rm; ++i) {
          for (int j = 0; j < _rn; ++j) { c[i * ldc + j] += bias[j]; }
        }
      }
      break;
    }
  }

#undef KERNEL
}
__MLLM_UNSAFE_OPT_END
}  // namespace

bool __mllm_blas_sgemm_nt_nt(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda, const float* B, int64_t ldb,
                             float* C, int64_t ldc, int ith, int thread_count) {
  if (m <= 0 || n <= 0 || k <= 0) return false;
  if (lda < k || ldb < n || ldc < n) return false;
  if (thread_count <= 0 || ith < 0 || ith >= thread_count) return false;
  /* dynamic tiling */
  int64_t mc = 8, nc = 16;
  if (m < 8) mc = 4;
  if (m < 4) mc = 1;
  if (n < 16) nc = 4;
  if (n < 4) nc = 1;
  int64_t yt = (m + mc - 1) / mc, xt = (n + nc - 1) / nc;
  int64_t tiles = yt * xt;
  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, job, 0, tiles, 1, {
    int64_t ii = (job / xt) * mc;
    int64_t jj = (job % xt) * nc;
    int64_t rm = std::min(mc, m - ii);
    int64_t rn = std::min(nc, n - jj);
    dispatch_tile(rm, rn, &A[ii * lda], lda, &B[jj], ldb, &C[ii * ldc + jj], ldc, k);
  });
  return true;
}

bool __mllm_blas_sgemm_nt_t(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda, const float* B, int64_t ldb, float* C,
                            int64_t ldc, int ith, const float* bias, int thread_count) {
  if (m <= 0 || n <= 0 || k <= 0) return false;
  if (lda < k || ldb < k || ldc < n) return false;
  if (thread_count <= 0 || ith < 0 || ith >= thread_count) return false;
  /* dynamic tiling */
  int64_t mc = 8, nc = 16;
  if (m < 8) mc = 4;
  if (m < 4) mc = 1;
  if (n < 16) nc = 4;
  if (n < 4) nc = 1;
  int64_t yt = (m + mc - 1) / mc, xt = (n + nc - 1) / nc;
  int64_t tiles = yt * xt;
  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, job, 0, tiles, 1, {
    int64_t ii = (job / xt) * mc;
    int64_t jj = (job % xt) * nc;
    int64_t rm = std::min(mc, m - ii);
    int64_t rn = std::min(nc, n - jj);
    dispatch_tile_nt_t(rm, rn, &A[ii * lda], lda, &B[jj * ldb], ldb, &C[ii * ldc + jj], ldc, k, bias ? &bias[jj] : bias);
  });
  return true;
}

void mllm_blas_matmul_fp32(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                           const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                           const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b, int thread_count) {
  // MxK, KxN
  if (!transpose_a && !transpose_b) {
    // gemv
    if (M == 1) {
      __mllm_blas_matmul_fp32_gemv(M, K, N, dst, A, B, C, transpose_a, transpose_b, thread_count);
    } else
    // gemm
    {
      if (C) { NYI("C not supported in mllm_blas_matmul_fp32::__mllm_blas_sgemm_nt_nt"); }
      __mllm_blas_sgemm_nt_nt(M, N, K, A, K, B, N, dst, N, 0, thread_count);
    }
    return;
  } else if (!transpose_a && transpose_b)
  // MxK, NxK
  {
    // gemv
    if (M == 1 && K % 32 == 0) {
      __mllm_blas_matmul_fp32_gemv(M, K, N, dst, A, B, C, transpose_a, transpose_b, thread_count);
    } else
    // gemm
    {
      __mllm_blas_sgemm_nt_t(M, N, K, A, K, B, K, dst, N, 0, C, thread_count);
    }
    return;
  } else {
    NYI("transpose_a && transpose_b not supported not supported in mllm_blas_matmul_fp32 gemm/gemv");
  }
}

void mllm_blas_batch_matmul_fp32(const int BATCH, const int M, const int K, const int N, const int Dst_batch_stride,
                                 const int A_batch_stride, const int B_batch_stride, const int C_batch_stride,
                                 mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                 const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                 bool transpose_b, int thread_count) {
  // MxK, KxN
  if (!transpose_a && !transpose_b) {
    // gemv
    if (M == 1) {
      __mllm_blas_batch_matmul_fp32_gemv(BATCH, M, K, N, Dst_batch_stride, A_batch_stride, B_batch_stride, C_batch_stride, dst,
                                         A, B, C, transpose_a, transpose_b, thread_count);
    } else
    // gemm
    {
      if (C) { NYI("C not supported in mllm_blas_batch_matmul_fp32::__mllm_blas_sgemm_nt_nt"); }
      // Parallel is in the inner loops, not here.
      for (int i = 0; i < BATCH; ++i) {
        __mllm_blas_sgemm_nt_nt(M, N, K, A + i * A_batch_stride, K, B + i * B_batch_stride, N, dst + i * Dst_batch_stride, N, 0,
                                thread_count);
      }
    }
    return;
  } else if (!transpose_a && transpose_b)
  // MxK, NxK
  {
    // gemv
    if (M == 1 && K % 32 == 0) {
      __mllm_blas_batch_matmul_fp32_gemv(BATCH, M, K, N, Dst_batch_stride, A_batch_stride, B_batch_stride, C_batch_stride, dst,
                                         A, B, C, transpose_a, transpose_b, thread_count);
    } else
    // gemm
    {
      // Parallel is in the inner loops, not here.
      for (int i = 0; i < BATCH; ++i) {
        __mllm_blas_sgemm_nt_t(M, N, K, A + i * A_batch_stride, K, B + i * B_batch_stride, K, dst + i * Dst_batch_stride, N, 0,
                               C, thread_count);
      }
    }
    return;
  } else {
    NYI("transpose_a && transpose_b not supported not supported in mllm_blas_matmul_fp32 gemm/gemv");
  }
}

}  // namespace mllm::cpu::arm
