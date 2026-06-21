// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cassert>

#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/backends/cpu/kernels/x86/mllm_blas/mllm_blas_sgemm.hpp"

namespace mllm::cpu::x86 {

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

  for (int s = 0; s < S; ++s) {
    dst[s] = C ? C[s] : 0.0f;
    for (int d = 0; d < D; ++d) { dst[s] += A[d] * B[s * D + d]; }
  }
}

// Optimized for decoding using AVX/AVX2.
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

#if defined(MLLM_HOST_FEATURE_AVX) || defined(MLLM_HOST_FEATURE_AVX2)
  // AVX processes 8 floats at a time
  const int DTileSize = 32;
  const int DTileCount = D / DTileSize;
  const int DRemainder = D % DTileSize;

  for (int s = 0; s < S; ++s) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const int s_offset = s * D;

    for (int d = 0; d < DTileCount; ++d) {
      const int d_offset = d * DTileSize;

      __m256 a0 = _mm256_loadu_ps(A + d_offset);
      __m256 b0 = _mm256_loadu_ps(B + s_offset + d_offset);
#if defined(MLLM_HOST_FEATURE_FMA)
      acc0 = _mm256_fmadd_ps(a0, b0, acc0);
#else
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
#endif

      __m256 a1 = _mm256_loadu_ps(A + d_offset + 8);
      __m256 b1 = _mm256_loadu_ps(B + s_offset + d_offset + 8);
#if defined(MLLM_HOST_FEATURE_FMA)
      acc1 = _mm256_fmadd_ps(a1, b1, acc1);
#else
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1));
#endif

      __m256 a2 = _mm256_loadu_ps(A + d_offset + 16);
      __m256 b2 = _mm256_loadu_ps(B + s_offset + d_offset + 16);
#if defined(MLLM_HOST_FEATURE_FMA)
      acc2 = _mm256_fmadd_ps(a2, b2, acc2);
#else
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(a2, b2));
#endif

      __m256 a3 = _mm256_loadu_ps(A + d_offset + 24);
      __m256 b3 = _mm256_loadu_ps(B + s_offset + d_offset + 24);
#if defined(MLLM_HOST_FEATURE_FMA)
      acc3 = _mm256_fmadd_ps(a3, b3, acc3);
#else
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(a3, b3));
#endif
    }

    // Combine accumulators
    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum0123 = _mm256_add_ps(sum01, sum23);

    // Horizontal sum of __m256
    // sum0123 = [a0, a1, a2, a3, a4, a5, a6, a7]
    __m128 hi = _mm256_extractf128_ps(sum0123, 1);  // [a4, a5, a6, a7]
    __m128 lo = _mm256_castps256_ps128(sum0123);    // [a0, a1, a2, a3]
    __m128 sum128 = _mm_add_ps(lo, hi);             // [a0+a4, a1+a5, a2+a6, a3+a7]
    __m128 shuf = _mm_movehdup_ps(sum128);          // [a1+a5, a1+a5, a3+a7, a3+a7]
    __m128 sums = _mm_add_ps(sum128, shuf);         // [a0+a1+a4+a5, _, a2+a3+a6+a7, _]
    shuf = _mm_movehl_ps(shuf, sums);               // [a2+a3+a6+a7, _, _, _]
    sums = _mm_add_ss(sums, shuf);
    float result = _mm_cvtss_f32(sums);

    // Handle remainder
    int d_start = DTileCount * DTileSize;
    for (int d = d_start; d < D; ++d) { result += A[d] * B[s_offset + d]; }

    if (C) {
      dst[s] = result + C[s];
    } else {
      dst[s] = result;
    }
  }
#else
  // Fallback to baseline
  __mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline(M, K, N, dst, A, B, C, transpose_a, transpose_b, thread_count);
#endif
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

// Optimized for decoding using AVX/AVX2.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                                          const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                                          const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                                                          int thread_count) {
#if defined(MLLM_HOST_FEATURE_AVX) || defined(MLLM_HOST_FEATURE_AVX2)
  // Initialize dst with C or zeros
  if (C != nullptr) {
    int n = 0;
    for (; n <= N - 8; n += 8) {
      __m256 c_vec = _mm256_loadu_ps(C + n);
      _mm256_storeu_ps(dst + n, c_vec);
    }
    for (; n < N; ++n) { dst[n] = C[n]; }
  } else {
    int n = 0;
    for (; n <= N - 8; n += 8) { _mm256_storeu_ps(dst + n, _mm256_setzero_ps()); }
    for (; n < N; ++n) { dst[n] = 0.0f; }
  }

  int k = 0;
  for (; k <= K - 4; k += 4) {
    __m256 a_vec =
        _mm256_set_ps(A[k + 3], A[k + 3], A[k + 2], A[k + 2], A[k + 1], A[k + 1], A[k], A[k]);  // For broadcasting later
    float a0 = A[k + 0];
    float a1 = A[k + 1];
    float a2 = A[k + 2];
    float a3 = A[k + 3];

    int n = 0;
    for (; n <= N - 8; n += 8) {
      __m256 dst_vec = _mm256_loadu_ps(dst + n);

      __m256 b_vec0 = _mm256_loadu_ps(B + (k + 0) * N + n);
      __m256 b_vec1 = _mm256_loadu_ps(B + (k + 1) * N + n);
      __m256 b_vec2 = _mm256_loadu_ps(B + (k + 2) * N + n);
      __m256 b_vec3 = _mm256_loadu_ps(B + (k + 3) * N + n);

      __m256 a0_vec = _mm256_set1_ps(a0);
      __m256 a1_vec = _mm256_set1_ps(a1);
      __m256 a2_vec = _mm256_set1_ps(a2);
      __m256 a3_vec = _mm256_set1_ps(a3);

#if defined(MLLM_HOST_FEATURE_FMA)
      dst_vec = _mm256_fmadd_ps(b_vec0, a0_vec, dst_vec);
      dst_vec = _mm256_fmadd_ps(b_vec1, a1_vec, dst_vec);
      dst_vec = _mm256_fmadd_ps(b_vec2, a2_vec, dst_vec);
      dst_vec = _mm256_fmadd_ps(b_vec3, a3_vec, dst_vec);
#else
      dst_vec = _mm256_add_ps(dst_vec, _mm256_mul_ps(b_vec0, a0_vec));
      dst_vec = _mm256_add_ps(dst_vec, _mm256_mul_ps(b_vec1, a1_vec));
      dst_vec = _mm256_add_ps(dst_vec, _mm256_mul_ps(b_vec2, a2_vec));
      dst_vec = _mm256_add_ps(dst_vec, _mm256_mul_ps(b_vec3, a3_vec));
#endif

      _mm256_storeu_ps(dst + n, dst_vec);
    }

    // Handle remainder
    for (; n < N; ++n) {
      float sum = dst[n];
      sum += a0 * B[(k + 0) * N + n];
      sum += a1 * B[(k + 1) * N + n];
      sum += a2 * B[(k + 2) * N + n];
      sum += a3 * B[(k + 3) * N + n];
      dst[n] = sum;
    }
  }

  // Handle remaining k
  for (; k < K; ++k) {
    float a_val = A[k];
    __m256 a_vec = _mm256_set1_ps(a_val);

    int n = 0;
    for (; n <= N - 8; n += 8) {
      __m256 b_vec = _mm256_loadu_ps(B + k * N + n);
      __m256 dst_vec = _mm256_loadu_ps(dst + n);
#if defined(MLLM_HOST_FEATURE_FMA)
      dst_vec = _mm256_fmadd_ps(b_vec, a_vec, dst_vec);
#else
      dst_vec = _mm256_add_ps(dst_vec, _mm256_mul_ps(b_vec, a_vec));
#endif
      _mm256_storeu_ps(dst + n, dst_vec);
    }
    for (; n < N; ++n) { dst[n] += a_val * B[k * N + n]; }
  }
#else
  // Fallback to baseline
  __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv_baseline(M, K, N, dst, A, B, C, transpose_a, transpose_b, thread_count);
#endif
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
#if defined(MLLM_HOST_FEATURE_AVX) || defined(MLLM_HOST_FEATURE_AVX2)
#define KERNEL(__tile_m, __tile_n) \
  case (__tile_m << 8) | __tile_n: MicroKernel<__tile_m, __tile_n>::accumulate(a, lda, b, ldb, c, ldc, k); break;

  switch ((std::min(rm, 8) << 8) | std::min(rn, 8)) {
    // AVX optimized kernels
    KERNEL(8, 8)
    KERNEL(4, 8)
    KERNEL(1, 8)
    // General GEMV, M = 1, decode
    KERNEL(1, 1)
    KERNEL(1, 2)
    KERNEL(1, 3)
    KERNEL(1, 4)
    KERNEL(1, 5)
    KERNEL(1, 6)
    KERNEL(1, 7)
    // Compiler Optimized Kernel
    KERNEL(2, 2)
    KERNEL(2, 4)
    KERNEL(2, 6)
    KERNEL(2, 8)
    KERNEL(4, 2)
    KERNEL(4, 4)
    KERNEL(4, 6)
    default: {
      auto _rm = std::min(rm, 8);
      auto _rn = std::min(rn, 8);
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
#else
  // SSE or scalar fallback
  auto _rm = std::min(rm, 8);
  auto _rn = std::min(rn, 8);
  for (int i = 0; i < _rm; ++i) {
    for (int j = 0; j < _rn; ++j) { c[i * ldc + j] = 0; }
  }
  for (int64_t l = 0; l < k; ++l) {
    for (int i = 0; i < _rm; ++i) {
      const float ai = a[i * lda + l];
      for (int j = 0; j < _rn; ++j) { c[i * ldc + j] += ai * b[l * ldb + j]; }
    }
  }
#endif
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
static inline void dispatch_tile_nt_t(int rm, int rn, const float* a, int64_t lda, const float* b, int64_t ldb, float* c,
                                      int64_t ldc, int64_t k, const float* bias) {
#define KERNEL(__tile_m, __tile_n)                                                          \
  case (__tile_m << 8) | __tile_n:                                                          \
    MicroKernel_NT_T_Bias<__tile_m, __tile_n>::accumulate(a, lda, b, ldb, c, ldc, k, bias); \
    break;

  switch ((std::min(rm, 8) << 8) | std::min(rn, 8)) {
    // Compiler Optimized Kernel
    KERNEL(8, 8)
    KERNEL(4, 8)
    KERNEL(1, 8)
    // General GEMV, M = 1, decode
    KERNEL(1, 1)
    KERNEL(1, 2)
    KERNEL(1, 3)
    KERNEL(1, 4)
    KERNEL(1, 5)
    KERNEL(1, 6)
    KERNEL(1, 7)
    // Compiler Optimized Kernel
    KERNEL(2, 2)
    KERNEL(2, 4)
    KERNEL(2, 6)
    KERNEL(2, 8)
    KERNEL(4, 2)
    KERNEL(4, 4)
    KERNEL(4, 6)
    default: {
      auto _rm = std::min(rm, 8);
      auto _rn = std::min(rn, 8);
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

  // Dynamic tiling - use 8x8 for AVX (8 floats per register)
  int64_t mc = 8, nc = 8;
  if (m < 8) mc = 4;
  if (m < 4) mc = 1;
  if (n < 8) nc = 4;
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

  // Dynamic tiling - use 8x8 for AVX
  int64_t mc = 8, nc = 8;
  if (m < 8) mc = 4;
  if (m < 4) mc = 1;
  if (n < 8) nc = 4;
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
    if (M == 1) {
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
    if (M == 1) {
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

}  // namespace mllm::cpu::x86
