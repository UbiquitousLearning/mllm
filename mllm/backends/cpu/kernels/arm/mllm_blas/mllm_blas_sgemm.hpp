// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/UnsafeMacros.hpp"

namespace mllm::cpu::arm {

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
  // Do not use
  // # pragma omp parallel for if (thread_count > 1)
  //                              ^^^^^^^^^^^^^^^^^^
  // Some platform(OSX) will generate inefficient code in this case. Use template instead.
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
  // Do not use
  // # pragma omp parallel for if (thread_count > 1)
  //                              ^^^^^^^^^^^^^^^^^^
  // Some platform(OSX) will generate inefficient code in this case. Use template instead.
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

template<>
struct MicroKernel<8, 16> {
  static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc, int64_t k) {
    // FIXME:
    // Too many registers are used !!!
    float32x4_t acc00 = vdupq_n_f32(0.0f);
    float32x4_t acc01 = vdupq_n_f32(0.0f);
    float32x4_t acc02 = vdupq_n_f32(0.0f);
    float32x4_t acc03 = vdupq_n_f32(0.0f);
    float32x4_t acc10 = vdupq_n_f32(0.0f);
    float32x4_t acc11 = vdupq_n_f32(0.0f);
    float32x4_t acc12 = vdupq_n_f32(0.0f);
    float32x4_t acc13 = vdupq_n_f32(0.0f);
    float32x4_t acc20 = vdupq_n_f32(0.0f);
    float32x4_t acc21 = vdupq_n_f32(0.0f);
    float32x4_t acc22 = vdupq_n_f32(0.0f);
    float32x4_t acc23 = vdupq_n_f32(0.0f);
    float32x4_t acc30 = vdupq_n_f32(0.0f);
    float32x4_t acc31 = vdupq_n_f32(0.0f);
    float32x4_t acc32 = vdupq_n_f32(0.0f);
    float32x4_t acc33 = vdupq_n_f32(0.0f);
    float32x4_t acc40 = vdupq_n_f32(0.0f);
    float32x4_t acc41 = vdupq_n_f32(0.0f);
    float32x4_t acc42 = vdupq_n_f32(0.0f);
    float32x4_t acc43 = vdupq_n_f32(0.0f);
    float32x4_t acc50 = vdupq_n_f32(0.0f);
    float32x4_t acc51 = vdupq_n_f32(0.0f);
    float32x4_t acc52 = vdupq_n_f32(0.0f);
    float32x4_t acc53 = vdupq_n_f32(0.0f);
    float32x4_t acc60 = vdupq_n_f32(0.0f);
    float32x4_t acc61 = vdupq_n_f32(0.0f);
    float32x4_t acc62 = vdupq_n_f32(0.0f);
    float32x4_t acc63 = vdupq_n_f32(0.0f);
    float32x4_t acc70 = vdupq_n_f32(0.0f);
    float32x4_t acc71 = vdupq_n_f32(0.0f);
    float32x4_t acc72 = vdupq_n_f32(0.0f);
    float32x4_t acc73 = vdupq_n_f32(0.0f);

    const float* a0_ptr = a;
    const float* a1_ptr = a + lda;
    const float* a2_ptr = a + 2 * lda;
    const float* a3_ptr = a + 3 * lda;
    const float* a4_ptr = a + 4 * lda;
    const float* a5_ptr = a + 5 * lda;
    const float* a6_ptr = a + 6 * lda;
    const float* a7_ptr = a + 7 * lda;

    for (int64_t l = 0; l < k; ++l) {
      float a0 = a0_ptr[l];
      float a1 = a1_ptr[l];
      float a2 = a2_ptr[l];
      float a3 = a3_ptr[l];
      float a4 = a4_ptr[l];
      float a5 = a5_ptr[l];
      float a6 = a6_ptr[l];
      float a7 = a7_ptr[l];

      float32x4_t b0 = vld1q_f32(b + l * ldb);
      float32x4_t b1 = vld1q_f32(b + l * ldb + 4);
      float32x4_t b2 = vld1q_f32(b + l * ldb + 8);
      float32x4_t b3 = vld1q_f32(b + l * ldb + 12);

      acc00 = vfmaq_n_f32(acc00, b0, a0);
      acc01 = vfmaq_n_f32(acc01, b1, a0);
      acc02 = vfmaq_n_f32(acc02, b2, a0);
      acc03 = vfmaq_n_f32(acc03, b3, a0);

      acc10 = vfmaq_n_f32(acc10, b0, a1);
      acc11 = vfmaq_n_f32(acc11, b1, a1);
      acc12 = vfmaq_n_f32(acc12, b2, a1);
      acc13 = vfmaq_n_f32(acc13, b3, a1);

      acc20 = vfmaq_n_f32(acc20, b0, a2);
      acc21 = vfmaq_n_f32(acc21, b1, a2);
      acc22 = vfmaq_n_f32(acc22, b2, a2);
      acc23 = vfmaq_n_f32(acc23, b3, a2);

      acc30 = vfmaq_n_f32(acc30, b0, a3);
      acc31 = vfmaq_n_f32(acc31, b1, a3);
      acc32 = vfmaq_n_f32(acc32, b2, a3);
      acc33 = vfmaq_n_f32(acc33, b3, a3);

      acc40 = vfmaq_n_f32(acc40, b0, a4);
      acc41 = vfmaq_n_f32(acc41, b1, a4);
      acc42 = vfmaq_n_f32(acc42, b2, a4);
      acc43 = vfmaq_n_f32(acc43, b3, a4);

      acc50 = vfmaq_n_f32(acc50, b0, a5);
      acc51 = vfmaq_n_f32(acc51, b1, a5);
      acc52 = vfmaq_n_f32(acc52, b2, a5);
      acc53 = vfmaq_n_f32(acc53, b3, a5);

      acc60 = vfmaq_n_f32(acc60, b0, a6);
      acc61 = vfmaq_n_f32(acc61, b1, a6);
      acc62 = vfmaq_n_f32(acc62, b2, a6);
      acc63 = vfmaq_n_f32(acc63, b3, a6);

      acc70 = vfmaq_n_f32(acc70, b0, a7);
      acc71 = vfmaq_n_f32(acc71, b1, a7);
      acc72 = vfmaq_n_f32(acc72, b2, a7);
      acc73 = vfmaq_n_f32(acc73, b3, a7);
    }
    vst1q_f32(c + 0 * ldc, acc00);
    vst1q_f32(c + 0 * ldc + 4, acc01);
    vst1q_f32(c + 0 * ldc + 8, acc02);
    vst1q_f32(c + 0 * ldc + 12, acc03);

    vst1q_f32(c + 1 * ldc, acc10);
    vst1q_f32(c + 1 * ldc + 4, acc11);
    vst1q_f32(c + 1 * ldc + 8, acc12);
    vst1q_f32(c + 1 * ldc + 12, acc13);

    vst1q_f32(c + 2 * ldc, acc20);
    vst1q_f32(c + 2 * ldc + 4, acc21);
    vst1q_f32(c + 2 * ldc + 8, acc22);
    vst1q_f32(c + 2 * ldc + 12, acc23);

    vst1q_f32(c + 3 * ldc, acc30);
    vst1q_f32(c + 3 * ldc + 4, acc31);
    vst1q_f32(c + 3 * ldc + 8, acc32);
    vst1q_f32(c + 3 * ldc + 12, acc33);

    vst1q_f32(c + 4 * ldc, acc40);
    vst1q_f32(c + 4 * ldc + 4, acc41);
    vst1q_f32(c + 4 * ldc + 8, acc42);
    vst1q_f32(c + 4 * ldc + 12, acc43);

    vst1q_f32(c + 5 * ldc, acc50);
    vst1q_f32(c + 5 * ldc + 4, acc51);
    vst1q_f32(c + 5 * ldc + 8, acc52);
    vst1q_f32(c + 5 * ldc + 12, acc53);

    vst1q_f32(c + 6 * ldc, acc60);
    vst1q_f32(c + 6 * ldc + 4, acc61);
    vst1q_f32(c + 6 * ldc + 8, acc62);
    vst1q_f32(c + 6 * ldc + 12, acc63);

    vst1q_f32(c + 7 * ldc, acc70);
    vst1q_f32(c + 7 * ldc + 4, acc71);
    vst1q_f32(c + 7 * ldc + 8, acc72);
    vst1q_f32(c + 7 * ldc + 12, acc73);
  }
};

template<>
struct MicroKernel<4, 16> {
  static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc, int64_t k) {
    float32x4_t acc00 = vdupq_n_f32(0.0f);
    float32x4_t acc01 = vdupq_n_f32(0.0f);
    float32x4_t acc02 = vdupq_n_f32(0.0f);
    float32x4_t acc03 = vdupq_n_f32(0.0f);
    float32x4_t acc10 = vdupq_n_f32(0.0f);
    float32x4_t acc11 = vdupq_n_f32(0.0f);
    float32x4_t acc12 = vdupq_n_f32(0.0f);
    float32x4_t acc13 = vdupq_n_f32(0.0f);
    float32x4_t acc20 = vdupq_n_f32(0.0f);
    float32x4_t acc21 = vdupq_n_f32(0.0f);
    float32x4_t acc22 = vdupq_n_f32(0.0f);
    float32x4_t acc23 = vdupq_n_f32(0.0f);
    float32x4_t acc30 = vdupq_n_f32(0.0f);
    float32x4_t acc31 = vdupq_n_f32(0.0f);
    float32x4_t acc32 = vdupq_n_f32(0.0f);
    float32x4_t acc33 = vdupq_n_f32(0.0f);

    const float* a0_ptr = a;
    const float* a1_ptr = a + lda;
    const float* a2_ptr = a + 2 * lda;
    const float* a3_ptr = a + 3 * lda;

    for (int64_t l = 0; l < k; ++l) {
      float a0 = a0_ptr[l];
      float a1 = a1_ptr[l];
      float a2 = a2_ptr[l];
      float a3 = a3_ptr[l];

      float32x4_t b0 = vld1q_f32(b + l * ldb);
      float32x4_t b1 = vld1q_f32(b + l * ldb + 4);
      float32x4_t b2 = vld1q_f32(b + l * ldb + 8);
      float32x4_t b3 = vld1q_f32(b + l * ldb + 12);

      acc00 = vfmaq_n_f32(acc00, b0, a0);
      acc01 = vfmaq_n_f32(acc01, b1, a0);
      acc02 = vfmaq_n_f32(acc02, b2, a0);
      acc03 = vfmaq_n_f32(acc03, b3, a0);

      acc10 = vfmaq_n_f32(acc10, b0, a1);
      acc11 = vfmaq_n_f32(acc11, b1, a1);
      acc12 = vfmaq_n_f32(acc12, b2, a1);
      acc13 = vfmaq_n_f32(acc13, b3, a1);

      acc20 = vfmaq_n_f32(acc20, b0, a2);
      acc21 = vfmaq_n_f32(acc21, b1, a2);
      acc22 = vfmaq_n_f32(acc22, b2, a2);
      acc23 = vfmaq_n_f32(acc23, b3, a2);

      acc30 = vfmaq_n_f32(acc30, b0, a3);
      acc31 = vfmaq_n_f32(acc31, b1, a3);
      acc32 = vfmaq_n_f32(acc32, b2, a3);
      acc33 = vfmaq_n_f32(acc33, b3, a3);
    }

    vst1q_f32(c + 0 * ldc, acc00);
    vst1q_f32(c + 0 * ldc + 4, acc01);
    vst1q_f32(c + 0 * ldc + 8, acc02);
    vst1q_f32(c + 0 * ldc + 12, acc03);

    vst1q_f32(c + 1 * ldc, acc10);
    vst1q_f32(c + 1 * ldc + 4, acc11);
    vst1q_f32(c + 1 * ldc + 8, acc12);
    vst1q_f32(c + 1 * ldc + 12, acc13);

    vst1q_f32(c + 2 * ldc, acc20);
    vst1q_f32(c + 2 * ldc + 4, acc21);
    vst1q_f32(c + 2 * ldc + 8, acc22);
    vst1q_f32(c + 2 * ldc + 12, acc23);

    vst1q_f32(c + 3 * ldc, acc30);
    vst1q_f32(c + 3 * ldc + 4, acc31);
    vst1q_f32(c + 3 * ldc + 8, acc32);
    vst1q_f32(c + 3 * ldc + 12, acc33);
  }
};

template<>
struct MicroKernel<1, 4> {
  static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc, int64_t k) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int64_t l = 0; l < k; ++l) { acc = vfmaq_f32(acc, vld1q_f32(b + l * ldb), vdupq_n_f32(a[l])); }
    vst1q_f32(c, acc);
  }
};

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

// template<>
// struct MicroKernel_NT_T_Bias<8, 16> {
//   static inline void accumulate(const float* __restrict a, int64_t lda, const float* __restrict b, int64_t ldb,
//                                 float* __restrict c, int64_t ldc, int64_t k, const float* __restrict bias) noexcept {
//     // TODO
//   }
// };

// template<>
// struct MicroKernel_NT_T_Bias<4, 4> {
//   static inline void accumulate(const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc, int64_t k,
//                                 const float* bias) noexcept {
//     // TODO
//   }
// };

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

}  // namespace mllm::cpu::arm
