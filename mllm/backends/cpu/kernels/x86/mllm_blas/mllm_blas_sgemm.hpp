// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/UnsafeMacros.hpp"

namespace mllm::cpu::x86 {

// Optimized for decoding.
// Q: [1, D]
// K: [S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline(
    const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
    const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b, int thread_count);

// Optimized for decoding.
// Q: [1, D]
// K: [S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
void __mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                                         const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                                         const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                                                         int thread_count);

// Optimized for decoding.
// Q: [B, H, 1, D]
// K: [B, H, S, D]
// D is small in mllm's case(small language model).
// D=64, 96, 128 ...
template<bool __enable_thread = false>
void __mllm_blas_batch_matmul_fp32_gemv_nt_t_decode_small_d_qk(const int BATCH, const int M, const int K, const int N,
                                                               const int Dst_batch_stride, const int A_batch_stride,
                                                               const int B_batch_stride, const int C_batch_stride,
                                                               mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                                               const mllm_fp32_t* __restrict__ B,
                                                               const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                                               bool transpose_b, int thread_count) {
  if constexpr (__enable_thread) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(b, 0, BATCH, 1, thread_count) {
      auto a_ptr = A + b * A_batch_stride;
      auto b_ptr = B + b * B_batch_stride;
      auto c_ptr = C + b * C_batch_stride;
      auto d_ptr = dst + b * Dst_batch_stride;
      __mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk(M, K, N, d_ptr, a_ptr, b_ptr, c_ptr, transpose_a, transpose_b, 0);
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()
  } else {
    for (int b = 0; b < BATCH; ++b) {
      auto a_ptr = A + b * A_batch_stride;
      auto b_ptr = B + b * B_batch_stride;
      auto c_ptr = C + b * C_batch_stride;
      auto d_ptr = dst + b * Dst_batch_stride;
      __mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk(M, K, N, d_ptr, a_ptr, b_ptr, c_ptr, transpose_a, transpose_b, 0);
    }
  }
}

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
void __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv_baseline(
    const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
    const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b, int thread_count);

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
void __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                                          const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                                          const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
                                                          int thread_count);

// Optimized for decoding.
// W: [B, H, 1, S]
// V: [B, H, S, D]
// D is small in mllm's case(small language model).
template<bool __enable_thread = false>
void __mllm_blas_batch_matmul_fp32_gemv_nt_nt_decode_small_d_wv(
    const int BATCH, const int M, const int K, const int N, const int Dst_batch_stride, const int A_batch_stride,
    const int B_batch_stride, const int C_batch_stride, mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
    const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b,
    int thread_count) {
  if constexpr (__enable_thread) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(b, 0, BATCH, 1, thread_count) {
      auto a_ptr = A + b * A_batch_stride;
      auto b_ptr = B + b * B_batch_stride;
      auto c_ptr = C + b * C_batch_stride;
      auto d_ptr = dst + b * Dst_batch_stride;
      __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv(M, K, N, d_ptr, a_ptr, b_ptr, c_ptr, transpose_a, transpose_b, 0);
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()
  } else {
    for (int b = 0; b < BATCH; ++b) {
      auto a_ptr = A + b * A_batch_stride;
      auto b_ptr = B + b * B_batch_stride;
      auto c_ptr = C + b * C_batch_stride;
      auto d_ptr = dst + b * Dst_batch_stride;
      __mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv(M, K, N, d_ptr, a_ptr, b_ptr, c_ptr, transpose_a, transpose_b, 0);
    }
  }
}

void __mllm_blas_matmul_fp32_gemv(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                                  const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                                  const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b, int thread_count);

void __mllm_blas_batch_matmul_fp32_gemv(const int BATCH, const int M, const int K, const int N, const int Dst_batch_stride,
                                        const int A_batch_stride, const int B_batch_stride, const int C_batch_stride,
                                        mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                        const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                        bool transpose_b, int thread_count);

#ifdef __cplusplus
extern "C" {
#endif

// C = A * B   (row-major, FP32)
// A : mxk   B : kxn   C : mxn
// lda = k, ldb = n, ldc = n
bool __mllm_blas_sgemm_nt_nt(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda, const float* B, int64_t ldb,
                             float* C, int64_t ldc, int ith, int thread_count);

#ifdef __cplusplus
}
#endif

template<int RM, int RN>
struct MicroKernel;

#if defined(MLLM_HOST_FEATURE_AVX) || defined(MLLM_HOST_FEATURE_AVX2)
#include <immintrin.h>

// AVX/AVX2 optimized 8x8 micro-kernel
template<>
struct MicroKernel<8, 8> {
  static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc, int64_t k) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    const float* a0_ptr = a;
    const float* a1_ptr = a + lda;
    const float* a2_ptr = a + 2 * lda;
    const float* a3_ptr = a + 3 * lda;
    const float* a4_ptr = a + 4 * lda;
    const float* a5_ptr = a + 5 * lda;
    const float* a6_ptr = a + 6 * lda;
    const float* a7_ptr = a + 7 * lda;

    for (int64_t l = 0; l < k; ++l) {
      __m256 b_vec = _mm256_loadu_ps(b + l * ldb);

      __m256 a0_vec = _mm256_set1_ps(a0_ptr[l]);
      __m256 a1_vec = _mm256_set1_ps(a1_ptr[l]);
      __m256 a2_vec = _mm256_set1_ps(a2_ptr[l]);
      __m256 a3_vec = _mm256_set1_ps(a3_ptr[l]);
      __m256 a4_vec = _mm256_set1_ps(a4_ptr[l]);
      __m256 a5_vec = _mm256_set1_ps(a5_ptr[l]);
      __m256 a6_vec = _mm256_set1_ps(a6_ptr[l]);
      __m256 a7_vec = _mm256_set1_ps(a7_ptr[l]);

#if defined(MLLM_HOST_FEATURE_FMA)
      acc0 = _mm256_fmadd_ps(a0_vec, b_vec, acc0);
      acc1 = _mm256_fmadd_ps(a1_vec, b_vec, acc1);
      acc2 = _mm256_fmadd_ps(a2_vec, b_vec, acc2);
      acc3 = _mm256_fmadd_ps(a3_vec, b_vec, acc3);
      acc4 = _mm256_fmadd_ps(a4_vec, b_vec, acc4);
      acc5 = _mm256_fmadd_ps(a5_vec, b_vec, acc5);
      acc6 = _mm256_fmadd_ps(a6_vec, b_vec, acc6);
      acc7 = _mm256_fmadd_ps(a7_vec, b_vec, acc7);
#else
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0_vec, b_vec));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1_vec, b_vec));
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(a2_vec, b_vec));
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(a3_vec, b_vec));
      acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(a4_vec, b_vec));
      acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(a5_vec, b_vec));
      acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(a6_vec, b_vec));
      acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(a7_vec, b_vec));
#endif
    }

    _mm256_storeu_ps(c + 0 * ldc, acc0);
    _mm256_storeu_ps(c + 1 * ldc, acc1);
    _mm256_storeu_ps(c + 2 * ldc, acc2);
    _mm256_storeu_ps(c + 3 * ldc, acc3);
    _mm256_storeu_ps(c + 4 * ldc, acc4);
    _mm256_storeu_ps(c + 5 * ldc, acc5);
    _mm256_storeu_ps(c + 6 * ldc, acc6);
    _mm256_storeu_ps(c + 7 * ldc, acc7);
  }
};

// AVX/AVX2 optimized 4x8 micro-kernel
template<>
struct MicroKernel<4, 8> {
  static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc, int64_t k) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const float* a0_ptr = a;
    const float* a1_ptr = a + lda;
    const float* a2_ptr = a + 2 * lda;
    const float* a3_ptr = a + 3 * lda;

    for (int64_t l = 0; l < k; ++l) {
      __m256 b_vec = _mm256_loadu_ps(b + l * ldb);

      __m256 a0_vec = _mm256_set1_ps(a0_ptr[l]);
      __m256 a1_vec = _mm256_set1_ps(a1_ptr[l]);
      __m256 a2_vec = _mm256_set1_ps(a2_ptr[l]);
      __m256 a3_vec = _mm256_set1_ps(a3_ptr[l]);

#if defined(MLLM_HOST_FEATURE_FMA)
      acc0 = _mm256_fmadd_ps(a0_vec, b_vec, acc0);
      acc1 = _mm256_fmadd_ps(a1_vec, b_vec, acc1);
      acc2 = _mm256_fmadd_ps(a2_vec, b_vec, acc2);
      acc3 = _mm256_fmadd_ps(a3_vec, b_vec, acc3);
#else
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0_vec, b_vec));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1_vec, b_vec));
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(a2_vec, b_vec));
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(a3_vec, b_vec));
#endif
    }

    _mm256_storeu_ps(c + 0 * ldc, acc0);
    _mm256_storeu_ps(c + 1 * ldc, acc1);
    _mm256_storeu_ps(c + 2 * ldc, acc2);
    _mm256_storeu_ps(c + 3 * ldc, acc3);
  }
};

// AVX/AVX2 optimized 1x8 micro-kernel (GEMV)
template<>
struct MicroKernel<1, 8> {
  static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc, int64_t k) {
    __m256 acc = _mm256_setzero_ps();

    for (int64_t l = 0; l < k; ++l) {
      __m256 b_vec = _mm256_loadu_ps(b + l * ldb);
      __m256 a_vec = _mm256_set1_ps(a[l]);
#if defined(MLLM_HOST_FEATURE_FMA)
      acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
#else
      acc = _mm256_add_ps(acc, _mm256_mul_ps(a_vec, b_vec));
#endif
    }

    _mm256_storeu_ps(c, acc);
  }
};

#endif  // MLLM_HOST_FEATURE_AVX || MLLM_HOST_FEATURE_AVX2

// Generic fallback micro-kernel
template<int RM, int RN>
struct MicroKernel {
  __MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
  static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc,
                                int64_t k) noexcept {
    for (int i = 0; i < RM; ++i) {
      for (int j = 0; j < RN; ++j) { c[i * ldc + j] = 0; }
    }
    for (int64_t l = 0; l < k; ++l) {
      for (int i = 0; i < RM; ++i) {
        const float ai = a[i * lda + l];
        for (int j = 0; j < RN; ++j) { c[i * ldc + j] += ai * b[l * ldb + j]; }
      }
    }
  }
  __MLLM_UNSAFE_OPT_END
};

template<int RM, int RN>
struct MicroKernel_NT_T_Bias;

template<int RM, int RN>
struct MicroKernel_NT_T_Bias {
  static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc, int64_t k,
                                const float* bias) {
#pragma unroll
    for (int i = 0; i < RM; ++i) {
#pragma unroll
      for (int j = 0; j < RN; ++j) {
        float sum = 0.0f;
        for (int64_t l = 0; l < k; ++l) { sum += a[i * lda + l] * b[j * ldb + l]; }
        c[i * ldc + j] = sum;
      }
    }
    if (bias != nullptr) {
#pragma unroll
      for (int i = 0; i < RM; ++i) {
#pragma unroll
        for (int j = 0; j < RN; ++j) { c[i * ldc + j] += bias[j]; }
      }
    }
  }
};

bool __mllm_blas_sgemm_nt_t(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda, const float* B, int64_t ldb, float* C,
                            int64_t ldc, int ith, const float* bias, int thread_count);

void mllm_blas_matmul_fp32(const int M, const int K, const int N, mllm_fp32_t* __restrict__ dst,
                           const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                           const mllm_fp32_t* __restrict__ C, bool transpose_a, bool transpose_b, int thread_count);

void mllm_blas_batch_matmul_fp32(const int BATCH, const int M, const int K, const int N, const int Dst_batch_stride,
                                 const int A_batch_stride, const int B_batch_stride, const int C_batch_stride,
                                 mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ A,
                                 const mllm_fp32_t* __restrict__ B, const mllm_fp32_t* __restrict__ C, bool transpose_a,
                                 bool transpose_b, int thread_count);

}  // namespace mllm::cpu::x86
