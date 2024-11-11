/**
 * @file qt8_qt8_fp32_gemm.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-11-06
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifdef __ARM_NEON
#include "qt8_qt8_fp32_gemm.hpp"
#include <arm_neon.h>
#include <cstdlib>
#include <omp.h>

namespace mllm::armv8 {

// This function is dropped !!!
void qt8_qt8_fp32_gemv(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
                       bool transpose_b) {
    if (!transpose_b) {
        // Not Supported Yet.
        return;
    }

    // I tile K with 16 element per block. the 16 element will be load to 128bit vector.
    if (K % 16) abort();

    int32_t k_blocks = K / 16;
    int32_t n_blocks = N / 4;
    int32_t n_blocks_left = N % 4;
    float scale = SA * SB;

    for (int32_t n_block = 0; n_block < n_blocks; n_block++) {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B + 4 * K * n_block;
        auto c_ptr = (float *)C;

        // final accumulattor
        // float acc_line_normal_reg[4] = {0.f, 0.f, 0.f, 0.f};
        // float32x4_t acc_line = vmovq_n_f32(0);

        // accumulator
        int16x8x2_t acc_line_0 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};
        int16x8x2_t acc_line_1 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};
        int16x8x2_t acc_line_2 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};
        int16x8x2_t acc_line_3 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};

        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);
            int8x8_t r0_l = vget_low_s8(r0);
            int8x8_t r0_h = vget_high_s8(r0);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + 16 * k_block);
            int8x8_t n0_l = vget_low_s8(n0);
            int8x8_t n0_h = vget_high_s8(n0);
            acc_line_0.val[0] = vmlal_s8(acc_line_0.val[0], n0_h, r0_h);
            acc_line_0.val[1] = vmlal_s8(acc_line_0.val[1], n0_l, r0_l);

            // load from block, n = 1
            int8x16_t n1 = vld1q_s8(b_ptr + K + 16 * k_block);
            int8x8_t n1_l = vget_low_s8(n1);
            int8x8_t n1_h = vget_high_s8(n1);
            acc_line_1.val[0] = vmlal_s8(acc_line_1.val[0], n1_h, r0_h);
            acc_line_1.val[1] = vmlal_s8(acc_line_1.val[1], n1_l, r0_l);

            // load from block, n = 2
            int8x16_t n2 = vld1q_s8(b_ptr + 2 * K + 16 * k_block);
            int8x8_t n2_l = vget_low_s8(n2);
            int8x8_t n2_h = vget_high_s8(n2);
            acc_line_2.val[0] = vmlal_s8(acc_line_2.val[0], n2_h, r0_h);
            acc_line_2.val[1] = vmlal_s8(acc_line_2.val[1], n2_l, r0_l);

            // load from block, n = 3
            int8x16_t n3 = vld1q_s8(b_ptr + 3 * K + 16 * k_block);
            int8x8_t n3_l = vget_low_s8(n3);
            int8x8_t n3_h = vget_high_s8(n3);
            acc_line_3.val[0] = vmlal_s8(acc_line_3.val[0], n3_h, r0_h);
            acc_line_3.val[1] = vmlal_s8(acc_line_3.val[1], n3_l, r0_l);
        }

        // accumulate i16 vector to single i32 value. And turn it to float.
        int32x4_t acc_line_0_i32_sum_0 = vpaddlq_s16(acc_line_0.val[0]);
        int32x4_t acc_line_0_i32_sum_1 = vpaddlq_s16(acc_line_0.val[1]);
        int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0_i32_sum_0) + vaddvq_s32(acc_line_0_i32_sum_1);
        // acc_line = vsetq_lane_f32((float)acc_line_0_i32, acc_line, 0);
        *(c_ptr + 4 * n_block + 0) = (float)acc_line_0_i32 * scale;

        int32x4_t acc_line_1_i32_sum_0 = vpaddlq_s16(acc_line_1.val[0]);
        int32x4_t acc_line_1_i32_sum_1 = vpaddlq_s16(acc_line_1.val[1]);
        int32_t acc_line_1_i32 = vaddvq_s32(acc_line_1_i32_sum_0) + vaddvq_s32(acc_line_1_i32_sum_1);
        // acc_line = vsetq_lane_f32((float)acc_line_1_i32, acc_line, 1);
        *(c_ptr + 4 * n_block + 1) = (float)acc_line_1_i32 * scale;

        int32x4_t acc_line_2_i32_sum_0 = vpaddlq_s16(acc_line_2.val[0]);
        int32x4_t acc_line_2_i32_sum_1 = vpaddlq_s16(acc_line_2.val[1]);
        int32_t acc_line_2_i32 = vaddvq_s32(acc_line_2_i32_sum_0) + vaddvq_s32(acc_line_2_i32_sum_1);
        // acc_line = vsetq_lane_f32((float)acc_line_2_i32, acc_line, 2);
        *(c_ptr + 4 * n_block + 2) = (float)acc_line_2_i32 * scale;

        int32x4_t acc_line_3_i32_sum_0 = vpaddlq_s16(acc_line_3.val[0]);
        int32x4_t acc_line_3_i32_sum_1 = vpaddlq_s16(acc_line_3.val[1]);
        int32_t acc_line_3_i32 = vaddvq_s32(acc_line_3_i32_sum_0) + vaddvq_s32(acc_line_3_i32_sum_1);
        // acc_line = vsetq_lane_f32((float)acc_line_3_i32, acc_line, 3);
        *(c_ptr + 4 * n_block + 3) = (float)acc_line_3_i32 * scale;

        // scale it.
        // acc_line = vmulq_n_f32(acc_line, scale);

        // store
        // vst1q_f32(c_ptr + 4 * n_block, acc_line);
    }

    // perform vector dot one by one.
    for (int32_t n = 0; n < n_blocks_left; n++) {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B + 4 * (N / 4) * K;
        auto c_ptr = (float *)C + 4 * (N / 4);

        int16x8x2_t acc_line_0 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};

        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);
            int8x8_t r0_l = vget_low_s8(r0);
            int8x8_t r0_h = vget_high_s8(r0);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + n * K + 16 * k_block);
            int8x8_t n0_l = vget_low_s8(n0);
            int8x8_t n0_h = vget_high_s8(n0);
            acc_line_0.val[0] = vmlal_s8(acc_line_0.val[0], n0_h, r0_h);
            acc_line_0.val[1] = vmlal_s8(acc_line_0.val[1], n0_l, r0_l);
        }

        // accumulate i16 vector to single i32 value. And turn it to float.
        int32x4_t acc_line_0_i32_sum_0 = vpaddlq_s16(acc_line_0.val[0]);
        int32x4_t acc_line_0_i32_sum_1 = vpaddlq_s16(acc_line_0.val[1]);
        int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0_i32_sum_0) + vaddvq_s32(acc_line_0_i32_sum_1);

        // store
        *(c_ptr + n) = (float)acc_line_0_i32 * scale;
    }
}

void qt8_qt8_fp32_gemv_sdot(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
                            bool transpose_b) {
    if (!transpose_b) {
        // Not Supported Yet.
        return;
    }

    // I tile K with 16 element per block. the 16 element will be load to 128bit vector.
    if (K % 16) abort();

    int32_t k_blocks = K / 16;
    int32_t n_blocks = N / 4;
    int32_t n_blocks_left = N % 4;
    float scale = SA * SB;

    for (int32_t n_block = 0; n_block < n_blocks; n_block++) {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B + 4 * K * n_block;
        auto c_ptr = (float *)C;

        // accumulator
        int32x4_t acc_line_0 = vmovq_n_s32(0);
        int32x4_t acc_line_1 = vmovq_n_s32(0);
        int32x4_t acc_line_2 = vmovq_n_s32(0);
        int32x4_t acc_line_3 = vmovq_n_s32(0);

        // Only support v8.2+ with dotproduct enabled.
        // Using SDOT. The throughput will increase 4 times compared to fp32 version.
        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + 16 * k_block);
            acc_line_0 = vdotq_s32(acc_line_0, r0, n0);

            // load from block, n = 1
            int8x16_t n1 = vld1q_s8(b_ptr + K + 16 * k_block);
            acc_line_1 = vdotq_s32(acc_line_1, r0, n1);

            // load from block, n = 2
            int8x16_t n2 = vld1q_s8(b_ptr + 2 * K + 16 * k_block);
            acc_line_2 = vdotq_s32(acc_line_2, r0, n2);

            // load from block, n = 3
            int8x16_t n3 = vld1q_s8(b_ptr + 3 * K + 16 * k_block);
            acc_line_3 = vdotq_s32(acc_line_3, r0, n3);
        }

        // reduce all and save to c_ptr
        int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0);
        *(c_ptr + 4 * n_block + 0) = (float)acc_line_0_i32 * scale;

        int32_t acc_line_1_i32 = vaddvq_s32(acc_line_1);
        *(c_ptr + 4 * n_block + 1) = (float)acc_line_1_i32 * scale;

        int32_t acc_line_2_i32 = vaddvq_s32(acc_line_2);
        *(c_ptr + 4 * n_block + 2) = (float)acc_line_2_i32 * scale;

        int32_t acc_line_3_i32 = vaddvq_s32(acc_line_3);
        *(c_ptr + 4 * n_block + 3) = (float)acc_line_3_i32 * scale;
    }

    // perform vector dot one by one.
    for (int32_t n = 0; n < n_blocks_left; n++) {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B + 4 * (N / 4) * K;
        auto c_ptr = (float *)C + 4 * (N / 4);

        int32x4_t acc_line_0 = vmovq_n_s32(0);

        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + n * K + 16 * k_block);
            acc_line_0 = vdotq_s32(acc_line_0, r0, n0);
        }

        // accumulate i32 vector to single i32 value. And turn it to float.
        int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0);

        // store
        *(c_ptr + n) = (float)acc_line_0_i32 * scale;
    }
}

void qt8_qt8_fp32_kernel_4x4_sdot(void *A, void *B, void *C, int32_t N, int32_t K, float SA,
                                  float SB, bool transpose_b) {
    if (!transpose_b) {
        // Not Supported Yet.
        return;
    }

    int32_t k_blocks = K / 16;
    float scale = SA * SB;

    // 4 x K, K x 4.
    if (K % 16) abort();

    auto a_ptr = (int8_t *)A;
    auto b_ptr = (int8_t *)B;
    auto c_ptr = (float *)C;

    // accumulator should contain 16 values.
    int32x4x4_t acc_line_0 = {{vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0)}};
    int32x4x4_t acc_line_1 = {{vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0)}};
    int32x4x4_t acc_line_2 = {{vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0)}};
    int32x4x4_t acc_line_3 = {{vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0)}};

    // final accumulator
    int32_t acc_0[4] = {0, 0, 0, 0};
    int32_t acc_1[4] = {0, 0, 0, 0};
    int32_t acc_2[4] = {0, 0, 0, 0};
    int32_t acc_3[4] = {0, 0, 0, 0};

    for (int k_block = 0; k_block < k_blocks; k_block++) {
        // load 4 vector from A
        // a1
        int8x16_t a0 = vld1q_s8(a_ptr + 16 * k_block);
        int8x16_t a1 = vld1q_s8(a_ptr + 16 * k_block + K);
        int8x16_t a2 = vld1q_s8(a_ptr + 16 * k_block + 2 * K);
        int8x16_t a3 = vld1q_s8(a_ptr + 16 * k_block + 3 * K);

        // load 4 vector from B
        int8x16_t b0 = vld1q_s8(b_ptr + 16 * k_block);
        int8x16_t b1 = vld1q_s8(b_ptr + 16 * k_block + K);
        int8x16_t b2 = vld1q_s8(b_ptr + 16 * k_block + 2 * K);
        int8x16_t b3 = vld1q_s8(b_ptr + 16 * k_block + 3 * K);

        acc_line_0.val[0] = vdotq_s32(acc_line_0.val[0], a0, b0);
        acc_line_0.val[1] = vdotq_s32(acc_line_0.val[1], a0, b1);
        acc_line_0.val[2] = vdotq_s32(acc_line_0.val[2], a0, b2);
        acc_line_0.val[3] = vdotq_s32(acc_line_0.val[3], a0, b3);

        acc_line_1.val[0] = vdotq_s32(acc_line_1.val[0], a1, b0);
        acc_line_1.val[1] = vdotq_s32(acc_line_1.val[1], a1, b1);
        acc_line_1.val[2] = vdotq_s32(acc_line_1.val[2], a1, b2);
        acc_line_1.val[3] = vdotq_s32(acc_line_1.val[3], a1, b3);

        acc_line_2.val[0] = vdotq_s32(acc_line_2.val[0], a2, b0);
        acc_line_2.val[1] = vdotq_s32(acc_line_2.val[1], a2, b1);
        acc_line_2.val[2] = vdotq_s32(acc_line_2.val[2], a2, b2);
        acc_line_2.val[3] = vdotq_s32(acc_line_2.val[3], a2, b3);

        acc_line_3.val[0] = vdotq_s32(acc_line_3.val[0], a3, b0);
        acc_line_3.val[1] = vdotq_s32(acc_line_3.val[1], a3, b1);
        acc_line_3.val[2] = vdotq_s32(acc_line_3.val[2], a3, b2);
        acc_line_3.val[3] = vdotq_s32(acc_line_3.val[3], a3, b3);
    }

    acc_0[0] = vaddvq_s32(acc_line_0.val[0]);
    acc_0[1] = vaddvq_s32(acc_line_0.val[1]);
    acc_0[2] = vaddvq_s32(acc_line_0.val[2]);
    acc_0[3] = vaddvq_s32(acc_line_0.val[3]);
    int32x4_t acc_0_vec_i32 = vld1q_s32(acc_0);
    float32x4_t acc_0_vec_f32 = vcvtq_f32_s32(acc_0_vec_i32);
    float32x4_t acc_0_vec_final = vmulq_n_f32(acc_0_vec_f32, scale);
    vst1q_f32(c_ptr, acc_0_vec_final);

    acc_1[0] = vaddvq_s32(acc_line_1.val[0]);
    acc_1[1] = vaddvq_s32(acc_line_1.val[1]);
    acc_1[2] = vaddvq_s32(acc_line_1.val[2]);
    acc_1[3] = vaddvq_s32(acc_line_1.val[3]);
    int32x4_t acc_1_vec_i32 = vld1q_s32(acc_1);
    float32x4_t acc_1_vec_f32 = vcvtq_f32_s32(acc_1_vec_i32);
    float32x4_t acc_1_vec_final = vmulq_n_f32(acc_1_vec_f32, scale);
    vst1q_f32(c_ptr + N, acc_1_vec_final);

    acc_2[0] = vaddvq_s32(acc_line_2.val[0]);
    acc_2[1] = vaddvq_s32(acc_line_2.val[1]);
    acc_2[2] = vaddvq_s32(acc_line_2.val[2]);
    acc_2[3] = vaddvq_s32(acc_line_2.val[3]);
    int32x4_t acc_2_vec_i32 = vld1q_s32(acc_2);
    float32x4_t acc_2_vec_f32 = vcvtq_f32_s32(acc_2_vec_i32);
    float32x4_t acc_2_vec_final = vmulq_n_f32(acc_2_vec_f32, scale);
    vst1q_f32(c_ptr + 2 * N, acc_2_vec_final);

    acc_3[0] = vaddvq_s32(acc_line_3.val[0]);
    acc_3[1] = vaddvq_s32(acc_line_3.val[1]);
    acc_3[2] = vaddvq_s32(acc_line_3.val[2]);
    acc_3[3] = vaddvq_s32(acc_line_3.val[3]);
    int32x4_t acc_3_vec_i32 = vld1q_s32(acc_3);
    float32x4_t acc_3_vec_f32 = vcvtq_f32_s32(acc_3_vec_i32);
    float32x4_t acc_3_vec_final = vmulq_n_f32(acc_3_vec_f32, scale);
    vst1q_f32(c_ptr + 3 * N, acc_3_vec_final);
}

void qt8_qt8_fp32_vec_dot(void *A, void *B, void *C, int32_t K, float SA, float SB) {
    if (K % 16) abort();

    int32_t k_blocks = K / 16;
    float scale = SA * SB;

    auto a_ptr = (int8_t *)A;
    auto b_ptr = (int8_t *)B;
    auto c_ptr = (float *)C;

    // accumulator
    int32x4_t acc_line_0 = vmovq_n_s32(0);

    // Only support v8.2+ with dotproduct enabled.
    // Using SDOT. The throughput will increase 4 times compared to fp32 version.
    for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
        // load from A
        int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);

        // load from block, n = 0
        int8x16_t n0 = vld1q_s8(b_ptr + 16 * k_block);
        acc_line_0 = vdotq_s32(acc_line_0, r0, n0);
    }

    // reduce all and save to c_ptr
    int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0);
    *(c_ptr) = (float)acc_line_0_i32 * scale;
}

// This function is dropped !!!
void qt8_qt8_fp32_gemm(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                       float SB, bool transpose_b) {
    if (M == 1) {
        qt8_qt8_fp32_gemv(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        // FIXME: tile in more efficient way.
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;
        for (int m = 0; m < M; ++m) {
            qt8_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
        }
    }
}

// This function is dropped !!!
void qt8_qt8_fp32_gemm_omp(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                           float SB, bool transpose_b) {
    if (M == 1) {
        qt8_qt8_fp32_gemv(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        // FIXME: tile in more efficient way.
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;
        if (M > 64) {
#pragma omp parallel for num_threads(4)
            for (int m = 0; m < M; ++m) {
                qt8_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
        } else {
            for (int m = 0; m < M; ++m) {
                qt8_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
        }
    }
}

void qt8_qt8_fp32_gemm_sdot(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                            float SB, bool transpose_b) {
    if (M == 1) {
        qt8_qt8_fp32_gemv_sdot(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;

        int32_t m_blocks = M / 4;
        int32_t n_blocks = N / 4;
        int32_t m_left = M % 4;
        int32_t n_left = N % 4;

        if (M < 4 || N < 4) {
            for (int m = 0; m < M; ++m) {
                qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
            return;
        }

        // main loop
        for (int m = 0; m < m_blocks; ++m) {
            for (int n = 0; n < n_blocks; ++n) {
                qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + 4 * n * K,
                                             c_ptr + 4 * m * N + 4 * n, N, K, SA, SB, transpose_b);
            }

            // some re cauculating may be needed
            if (n_left) {
                qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                             c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA, SB,
                                             transpose_b);
            }
        }

        if (m_left) {
            for (int m = m_blocks * 4; m < M; ++m) {
                qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
        }
    }
}

void qt8_qt8_fp32_gemm_sdot_omp(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K,
                                float SA, float SB, bool transpose_b) {
    if (M == 1) {
        qt8_qt8_fp32_gemv_sdot(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;

        int32_t m_blocks = M / 4;
        int32_t n_blocks = N / 4;
        int32_t m_left = M % 4;
        int32_t n_left = N % 4;

        if (M < 4 || N < 4) {
            for (int m = 0; m < M; ++m) {
                qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
            return;
        }

        if (M > 64) {
            // main loop
#pragma omp parallel for num_threads(4)
            for (int m = 0; m < m_blocks; ++m) {
                for (int n = 0; n < n_blocks; ++n) {
                    qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + 4 * n * K,
                                                 c_ptr + 4 * m * N + 4 * n, N, K, SA, SB, transpose_b);
                }

                // some re cauculating may be needed
                if (n_left) {
                    qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                                 c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA,
                                                 SB, transpose_b);
                }
            }

            if (m_left) {
                for (int m = m_blocks * 4; m < M; ++m) {
                    qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
                }
            }
        } else {
            // main loop
            for (int m = 0; m < m_blocks; ++m) {
                for (int n = 0; n < n_blocks; ++n) {
                    qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + 4 * n * K,
                                                 c_ptr + 4 * m * N + 4 * n, N, K, SA, SB, transpose_b);
                }

                // some re cauculating may be needed
                if (n_left) {
                    qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                                 c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA,
                                                 SB, transpose_b);
                }
            }

            if (m_left) {
                for (int m = m_blocks * 4; m < M; ++m) {
                    qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
                }
            }
        }
    }
}
} // namespace mllm::armv8
#endif