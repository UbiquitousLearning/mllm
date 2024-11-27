//
// Created by Rongjie Yi on 23-10-24.
//

#ifndef MLLM_MATMUL_HPP
#define MLLM_MATMUL_HPP

#include "VecDot.hpp"
using namespace mllm;

ErrorCode mat_mul(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = true, int thread_count = 4);

ErrorCode mat_mul_i8(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false, int thread_count = 4, float scale1 = 1.0f, float scale2 = 1.0f);

#ifdef __ARM_NEON

#ifndef __ARM_NEON
#error \
    "The mllm-advance Armv8 backend is enbaled but __ARM_NEON is not defined. Pls use cross-compile toolchains(such as NDK) to compile."
#endif

#include <cstdint>

namespace mllm::armv8 {

/**
 * @brief Decoding stage. qt8_qt8_fp32_gemm will accpect q(1 x K), k^T(K x N) as inputs. GEMV Like.
 *
 * This function is dropped !!!
 *
 * @param A
 * @param B
 * @param C
 * @param N
 * @param K
 * @param SA
 * @param SB
 * @param transpose_b
 */
[[deprecated]] void qt8_qt8_fp32_gemv(void *A, void *B, void *C, int32_t N, int32_t K, float SA,
                                      float SB, bool transpose_b = false);

/**
 * @brief Same logic as qt8_qt8_fp32_gemv. But accumulate numbers into int32 accumulator.
 *
 * @param A
 * @param B
 * @param C
 * @param N
 * @param K
 * @param SA
 * @param SB
 * @param transpose_b
 */
void qt8_qt8_fp32_gemv_sdot(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
                            bool transpose_b = false);

/**
 * @brief
 *
 * @param A
 * @param B
 * @param C
 * @param K
 * @param SA
 * @param SB
 * @param transpose_b
 */
void qt8_qt8_fp32_kernel_4x4_sdot(void *A, void *B, void *C, int32_t N, int32_t K, float SA,
                                  float SB, bool transpose_b = false);

/**
 * @brief Per-Tensor Quantized Int8 vec dot product.
 *
 * @param A
 * @param B
 * @param C
 * @param K
 * @param SA
 * @param SB
 */
void qt8_qt8_fp32_vec_dot(void *A, void *B, void *C, int32_t K, float SA, float SB);

/**
 * @brief Per-Tensor Quantized Int8 GEMM.
 * A(per-tensor signed int 8) @ B(per-tensor signed int 8) -> C(fp32)
 * A(M x K), B(K x N), C(M x K)
 *
 * This function is dropped !!!
 *
 * @param A int8_t array
 * @param B int8_t array
 * @param C float array
 * @param M
 * @param N
 * @param K
 * @param SA Per-tensor scale for A
 * @param SB Per-tensor scale for B
 */
[[deprecated]] void qt8_qt8_fp32_gemm(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K,
                                      float SA, float SB, bool transpose_b = false);

/**
 * @brief Using openmp on GEMM. GEMV always disable multithread.
 *
 * This function is dropped !!!
 *
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @param SA
 * @param SB
 * @param transpose_b
 */
[[deprecated]] void qt8_qt8_fp32_gemm_omp(void *A, void *B, void *C, int32_t M, int32_t N,
                                          int32_t K, float SA, float SB, bool transpose_b = false);

/**
 * @brief Per-Tensor Quantized Int8 GEMM.
 * A(per-tensor signed int 8) @ B(per-tensor signed int 8) -> C(fp32)
 * A(M x K), B(K x N), C(M x K)
 *
 * @param A int8_t array
 * @param B int8_t array
 * @param C float array
 * @param M
 * @param N
 * @param K
 * @param SA Per-tensor scale for A
 * @param SB Per-tensor scale for B
 */
void qt8_qt8_fp32_gemm_sdot(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                            float SB, bool transpose_b = false);

/**
 * @brief Using openmp on GEMM. GEMV always disable multithread.
 *
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @param SA
 * @param SB
 * @param transpose_b
 */
void qt8_qt8_fp32_gemm_sdot_omp(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K,
                                float SA, float SB, bool transpose_b = false);

} // namespace mllm::armv8
#endif

#endif // MLLM_MATMUL_HPP
