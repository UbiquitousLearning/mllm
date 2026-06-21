// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/vec_dot.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/vec_dot_type.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q2.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q3.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q4.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q6.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q8.hpp"

// Helper macros
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Conversion macros - use existing one if already defined
#ifndef MLLM_FP16_TO_FP32
#define MLLM_FP16_TO_FP32(x) ((float)(x))
#endif

// Namespace for template specializations
namespace mllm::cpu::ggml {

// Helper function for Q4_K quantization
static inline void get_scale_min_k4(int j, const uint8_t* __restrict q, uint8_t* __restrict d, uint8_t* __restrict m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Add function implementations inside namespace
void fp32_add_row_to(int n, const float* MLLM_RESTRICT src, float* MLLM_RESTRICT dst, float alpha) {
  int i = 0;
#ifdef __AVX2__
  __m256 alpha_vec = _mm256_set1_ps(alpha);  // load alpha into 8 float register
  for (; i <= n - 8; i += 8) {
    __m256 src_vec = _mm256_loadu_ps(src + i);                      // load 8 float from src
    __m256 dst_vec = _mm256_loadu_ps(dst + i);                      // load 8 float from dst
    __m256 res_vec = _mm256_fmadd_ps(src_vec, alpha_vec, dst_vec);  // alpha * src + dst
    _mm256_storeu_ps(dst + i, res_vec);                             // store back to dst
  }
#elif defined(__ARM_NEON)
  float32x4_t alpha_vec = vdupq_n_f32(alpha);  // load alpha into all elements of a 128-bit register

  // Main loop for multiples of 4
  for (; i <= n - 4; i += 4) {
    float32x4_t src_vec = vld1q_f32(src + i);
    float32x4_t dst_vec = vld1q_f32(dst + i);
    float32x4_t res_vec = vmlaq_f32(dst_vec, src_vec, alpha_vec);  // calculate alpha * src + dst
    vst1q_f32(dst + i, res_vec);                                   // store result back to dst
  }
#endif
  for (; i < n; ++i) { dst[i] = dst[i] + alpha * src[i]; }
}

void fp_16_add_row_to(int n, const mllm::mllm_fp16_t* MLLM_RESTRICT src, float* MLLM_RESTRICT dst, float alpha) {
  int i = 0;
#ifdef __AVX2__
  __m256 alpha_vec = _mm256_set1_ps(alpha);  // load alpha into 8 float register
  for (; i <= n - 8; i += 8) {
    __m128i src_fp16 = _mm_loadu_si128((__m128i const*)(src + i));  // load 8 fp16 from src
    __m256 src_vec = _mm256_cvtph_ps(src_fp16);                     // convert to 8 fp32
    __m256 dst_vec = _mm256_loadu_ps(dst + i);                      // load 8 float from dst
    __m256 res_vec = _mm256_fmadd_ps(src_vec, alpha_vec, dst_vec);  // alpha * src + dst
    _mm256_storeu_ps(dst + i, res_vec);                             // store back to dst
  }
#elif defined(__ARM_NEON)
  MLLM_ERROR_EXIT(ExitCode::kCoreError, "not support now");
#endif
  for (; i < n; ++i) { dst[i] = dst[i] + alpha * MLLM_FP16_TO_FP32(src[i]); }
}

void q4_0_add_row_to(int n, const mllm::block_q4_0* MLLM_RESTRICT src, float* MLLM_RESTRICT dst, float alpha) {
  assert(n % QK4_0 == 0);
  auto num_blocks = n / QK4_0;

  int i = 0;
#ifdef __AVX2__
  // TODO: not implemented
#elif defined(__ARM_NEON)
  // TODO: not implemented
#endif

  // Process the remaining elements
  for (; i < num_blocks; ++i) {
    auto scale = MLLM_FP16_TO_FP32(src[i].d) * alpha;
    auto offset = i * QK4_0;

    for (int j = 0; j < QK4_0 / 2; ++j) {
      const int v0 = (src[i].qs[j] & 0x0F) - 8;
      const int v1 = (src[i].qs[j] >> 4) - 8;
      dst[offset + j] = dst[offset + j] + (scale * static_cast<float>(v0));
      dst[offset + j + QK4_0 / 2] = dst[offset + j + QK4_0 / 2] + (scale * static_cast<float>(v1));
    }
  }
}

void q4_k_add_row_to(int n, const mllm::block_q4_K* MLLM_RESTRICT src, float* MLLM_RESTRICT dst, float alpha) {
  assert(n % QK_K == 0);
  assert(QK_K == 256);  // TODO: It is wired here for now
  const int nb = n / QK_K;

  for (int i = 0; i < nb; i++) {
    const uint8_t* q = src[i].qs;

    const float d = MLLM_FP16_TO_FP32(src[i].d);       // scale for super block's d
    const float min = MLLM_FP16_TO_FP32(src[i].dmin);  // scale for super block's min

    int is = 0;
    uint8_t sc;
    uint8_t m;
    for (int j = 0; j < QK_K; j += 64) {
      get_scale_min_k4(is + 0, src[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, src[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) {
        *dst = *dst + (d1 * (q[l] & 0xF) - m1) * alpha;
        dst++;
      }
      for (int l = 0; l < 32; ++l) {
        *dst = *dst + (d2 * (q[l] >> 4) - m2) * alpha;
        dst++;
      }
      q += 32;
      is += 2;
    }
  }
}

void q6_k_add_row_to(int n, const mllm::block_q6_K* MLLM_RESTRICT src, float* MLLM_RESTRICT dst, float alpha) {
  assert(n % QK_K == 0);
  const int nb = n / QK_K;

  for (int i = 0; i < nb; i++) {
    const float d = MLLM_FP16_TO_FP32(src[i].d);
    const float scale = d * alpha;

    const uint8_t* __restrict ql = src[i].ql;
    const uint8_t* __restrict qh = src[i].qh;
    const int8_t* __restrict sc = src[i].scales;

    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        dst[l + 0] += scale * sc[is + 0] * q1;
        dst[l + 32] += scale * sc[is + 2] * q2;
        dst[l + 64] += scale * sc[is + 4] * q3;
        dst[l + 96] += scale * sc[is + 6] * q4;
      }
      dst += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

void q8_0_add_row_to(int n, const mllm::block_q8_0* MLLM_RESTRICT src, float* MLLM_RESTRICT dst, float alpha) {
  static const int qk = QK8_0;

  assert(n % qk == 0);

  const int nb = n / qk;

  const mllm::block_q8_0* __restrict x = src;

  for (int i = 0; i < nb; i++) {
    const float scale = MLLM_FP16_TO_FP32(x[i].d) * alpha;

    for (int j = 0; j < qk; ++j) { dst[i * qk + j] += x[i].qs[j] * scale; }
  }
}

void q8_k_add_row_to(int n, const mllm::block_q8_K* MLLM_RESTRICT src, float* MLLM_RESTRICT dst, float alpha) {
  assert(n % QK_K == 0);
  const int nb = n / QK_K;

  for (int i = 0; i < nb; i++) {
    auto scale = src[i].d * alpha;
    for (int j = 0; j < QK_K; ++j) { dst[j] += scale * src[i].qs[j]; }
    dst += QK_K;
  }
}

// Template specializations for the old template design
const size_t TypeTraits<MLLM_TYPE_F32>::size = sizeof(float);
const int TypeTraits<MLLM_TYPE_F32>::blck_size = 1;
const int TypeTraits<MLLM_TYPE_F32>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_F32>::to_float = nullptr;
const mllm_from_float_func TypeTraits<MLLM_TYPE_F32>::from_float = nullptr;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_F32>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_F32>::vec_dot = (mllm_vec_dot_func)vec_dot_fp32;
const DataTypes TypeTraits<MLLM_TYPE_F32>::vec_dot_type = MLLM_TYPE_F32;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_F32>::add_row_to = (mllm_vec_add_row_func)fp32_add_row_to;
const mllm_gemv_func TypeTraits<MLLM_TYPE_F32>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_F32>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_F16>::size = sizeof(mllm::mllm_fp16_t);
const int TypeTraits<MLLM_TYPE_F16>::blck_size = 1;
const int TypeTraits<MLLM_TYPE_F16>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_F16>::to_float = nullptr;
const mllm_from_float_func TypeTraits<MLLM_TYPE_F16>::from_float = nullptr;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_F16>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_F16>::vec_dot = (mllm_vec_dot_func)vec_dot_fp16;
const DataTypes TypeTraits<MLLM_TYPE_F16>::vec_dot_type = MLLM_TYPE_F16;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_F16>::add_row_to = (mllm_vec_add_row_func)fp_16_add_row_to;
const mllm_gemv_func TypeTraits<MLLM_TYPE_F16>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_F16>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_Q4_0>::size = sizeof(mllm::block_q4_0);
const int TypeTraits<MLLM_TYPE_Q4_0>::blck_size = QK4_0;
const int TypeTraits<MLLM_TYPE_Q4_0>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_Q4_0>::to_float = (mllm_to_float_func)dequantize_row_q4_0;
const mllm_from_float_func TypeTraits<MLLM_TYPE_Q4_0>::from_float = (mllm_from_float_func)quantize_row_q4_0;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_Q4_0>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_Q4_0>::vec_dot = (mllm_vec_dot_func)vec_dot_q4_0_q8_0;
const DataTypes TypeTraits<MLLM_TYPE_Q4_0>::vec_dot_type = MLLM_TYPE_Q8_0;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_Q4_0>::add_row_to = (mllm_vec_add_row_func)q4_0_add_row_to;
const mllm_gemv_func TypeTraits<MLLM_TYPE_Q4_0>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_Q4_0>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_Q4_K>::size = sizeof(mllm::block_q4_K);
const int TypeTraits<MLLM_TYPE_Q4_K>::blck_size = QK_K;
const int TypeTraits<MLLM_TYPE_Q4_K>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_Q4_K>::to_float = (mllm_to_float_func)dequantize_row_q4_K;
const mllm_from_float_func TypeTraits<MLLM_TYPE_Q4_K>::from_float = (mllm_from_float_func)quantize_row_q4_K;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_Q4_K>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_Q4_K>::vec_dot = (mllm_vec_dot_func)vec_dot_q4_K_q8_K;
const DataTypes TypeTraits<MLLM_TYPE_Q4_K>::vec_dot_type = MLLM_TYPE_Q8_K;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_Q4_K>::add_row_to = (mllm_vec_add_row_func)q4_k_add_row_to;
const mllm_gemv_func TypeTraits<MLLM_TYPE_Q4_K>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_Q4_K>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_Q6_K>::size = sizeof(mllm::block_q6_K);
const int TypeTraits<MLLM_TYPE_Q6_K>::blck_size = QK_K;
const int TypeTraits<MLLM_TYPE_Q6_K>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_Q6_K>::to_float = (mllm_to_float_func)dequantize_row_q6_K;
const mllm_from_float_func TypeTraits<MLLM_TYPE_Q6_K>::from_float = (mllm_from_float_func)quantize_row_q6_K;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_Q6_K>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_Q6_K>::vec_dot = (mllm_vec_dot_func)vec_dot_q6_K_q8_K;
const DataTypes TypeTraits<MLLM_TYPE_Q6_K>::vec_dot_type = MLLM_TYPE_Q8_K;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_Q6_K>::add_row_to = (mllm_vec_add_row_func)q6_k_add_row_to;
const mllm_gemv_func TypeTraits<MLLM_TYPE_Q6_K>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_Q6_K>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_Q8_0>::size = sizeof(mllm::block_q8_0);
const int TypeTraits<MLLM_TYPE_Q8_0>::blck_size = QK8_0;
const int TypeTraits<MLLM_TYPE_Q8_0>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_Q8_0>::to_float = (mllm_to_float_func)dequantize_row_q8_0;
const mllm_from_float_func TypeTraits<MLLM_TYPE_Q8_0>::from_float = (mllm_from_float_func)quantize_row_q8_0;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_Q8_0>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_Q8_0>::vec_dot = (mllm_vec_dot_func)vec_dot_q8_0_q8_0;
const DataTypes TypeTraits<MLLM_TYPE_Q8_0>::vec_dot_type = MLLM_TYPE_Q8_0;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_Q8_0>::add_row_to = (mllm_vec_add_row_func)q8_0_add_row_to;
const mllm_gemv_func TypeTraits<MLLM_TYPE_Q8_0>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_Q8_0>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_Q8_K>::size = sizeof(mllm::block_q8_K);
const int TypeTraits<MLLM_TYPE_Q8_K>::blck_size = QK_K;
const int TypeTraits<MLLM_TYPE_Q8_K>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_Q8_K>::to_float = (mllm_to_float_func)dequantize_row_q8_K;
const mllm_from_float_func TypeTraits<MLLM_TYPE_Q8_K>::from_float = (mllm_from_float_func)quantize_row_q8_K;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_Q8_K>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_Q8_K>::vec_dot = nullptr;  // Q8_K has no vec_dot function in old version
const DataTypes TypeTraits<MLLM_TYPE_Q8_K>::vec_dot_type = MLLM_TYPE_Q8_K;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_Q8_K>::add_row_to = (mllm_vec_add_row_func)q8_k_add_row_to;
const mllm_gemv_func TypeTraits<MLLM_TYPE_Q8_K>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_Q8_K>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_Q2_K>::size = sizeof(mllm::block_q2_K);
const int TypeTraits<MLLM_TYPE_Q2_K>::blck_size = QK_K;
const int TypeTraits<MLLM_TYPE_Q2_K>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_Q2_K>::to_float = (mllm_to_float_func)dequantize_row_q2_K;
const mllm_from_float_func TypeTraits<MLLM_TYPE_Q2_K>::from_float = (mllm_from_float_func)quantize_row_q2_K;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_Q2_K>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_Q2_K>::vec_dot = (mllm_vec_dot_func)vec_dot_q2_K_q8_K;
const DataTypes TypeTraits<MLLM_TYPE_Q2_K>::vec_dot_type = MLLM_TYPE_Q8_K;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_Q2_K>::add_row_to = nullptr;  // Not implemented
const mllm_gemv_func TypeTraits<MLLM_TYPE_Q2_K>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_Q2_K>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_Q3_K>::size = sizeof(mllm::block_q3_K);
const int TypeTraits<MLLM_TYPE_Q3_K>::blck_size = QK_K;
const int TypeTraits<MLLM_TYPE_Q3_K>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_Q3_K>::to_float = (mllm_to_float_func)dequantize_row_q3_K;
const mllm_from_float_func TypeTraits<MLLM_TYPE_Q3_K>::from_float = (mllm_from_float_func)quantize_row_q3_K;
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_Q3_K>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_Q3_K>::vec_dot = (mllm_vec_dot_func)vec_dot_q3_K_q8_K;
const DataTypes TypeTraits<MLLM_TYPE_Q3_K>::vec_dot_type = MLLM_TYPE_Q8_K;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_Q3_K>::add_row_to = nullptr;  // Not implemented
const mllm_gemv_func TypeTraits<MLLM_TYPE_Q3_K>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_Q3_K>::gemm = nullptr;

const size_t TypeTraits<MLLM_TYPE_IQ2_XXS>::size = sizeof(mllm::block_iq2_xxs);
const int TypeTraits<MLLM_TYPE_IQ2_XXS>::blck_size = QK_K;
const int TypeTraits<MLLM_TYPE_IQ2_XXS>::blck_size_interleave = 1;
const mllm_to_float_func TypeTraits<MLLM_TYPE_IQ2_XXS>::to_float = (mllm_to_float_func)dequantize_row_iq2_xxs;
const mllm_from_float_func TypeTraits<MLLM_TYPE_IQ2_XXS>::from_float = nullptr;  // quantize_row_iq2_xxs not found
const mllm_from_float_to_mat_func TypeTraits<MLLM_TYPE_IQ2_XXS>::from_float_to_mat = nullptr;
const mllm_vec_dot_func TypeTraits<MLLM_TYPE_IQ2_XXS>::vec_dot = (mllm_vec_dot_func)vec_dot_iq2_xxs_q8_K;
const DataTypes TypeTraits<MLLM_TYPE_IQ2_XXS>::vec_dot_type = MLLM_TYPE_Q8_K;
const mllm_vec_add_row_func TypeTraits<MLLM_TYPE_IQ2_XXS>::add_row_to = nullptr;  // Not implemented
const mllm_gemv_func TypeTraits<MLLM_TYPE_IQ2_XXS>::gemv = nullptr;
const mllm_gemm_func TypeTraits<MLLM_TYPE_IQ2_XXS>::gemm = nullptr;

}  // namespace mllm::cpu::ggml
