/**
 * @file fp32_qt8_fp32_gemm.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-11-08
 *
 * compile flag: -march=armv8.2-a+dotprod+fp16+fp16fml
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#ifdef __ARM_NEON

#include <arm_neon.h>

// A: fp32
// B: Int8
// C: fp32
// A->fp16, B->fp16, mla to fp32
namespace mllm::armv8 {

/**
 * @brief A: FP32, B, Int8, C: FP32
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
void fp32_qt8_fp32_gemv(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
                        bool transpose_b = false);

/**
 * @brief A: FP32, B, Int8, C: FP32
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
void fp32_qt8_fp32_kernel_4x4(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
                              bool transpose_b = false);

/**
 * @brief A: FP32, B, Int8, C: FP32
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
void fp32_qt8_fp32_gemm(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                        float SB, bool transpose_b = false);

/**
 * @brief A: FP32, B, Int8, C: FP32
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
void fp32_qt8_fp32_gemm_omp(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                            float SB, bool transpose_b = false);
} // namespace mllm::armv8
#endif
