#ifdef __ARM_NEON

#include "fp32_qt8_fp32_gemm.hpp"
#include <arm_neon.h>
#include <cstdlib>
#include <omp.h>

namespace mllm::armv8 {

void fp32_qt8_fp32_gemv(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
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
        auto a_ptr = (float *)A;
        auto b_ptr = (int8_t *)B + 4 * K * n_block;
        auto c_ptr = (float *)C;

        // accumulator
        float32x4x2_t acc_line_0;
        float32x4x2_t acc_line_1;
        float32x4x2_t acc_line_2;
        float32x4x2_t acc_line_3;
        acc_line_0.val[0] = vdupq_n_f32(0.f);
        acc_line_0.val[1] = vdupq_n_f32(0.f);
        acc_line_1.val[0] = vdupq_n_f32(0.f);
        acc_line_1.val[1] = vdupq_n_f32(0.f);
        acc_line_2.val[0] = vdupq_n_f32(0.f);
        acc_line_2.val[1] = vdupq_n_f32(0.f);
        acc_line_3.val[0] = vdupq_n_f32(0.f);
        acc_line_3.val[1] = vdupq_n_f32(0.f);

        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            float32x4x4_t r0;
            r0.val[0] = vld1q_f32(a_ptr + 16 * k_block);
            r0.val[1] = vld1q_f32(a_ptr + 16 * k_block + 4);
            r0.val[2] = vld1q_f32(a_ptr + 16 * k_block + 8);
            r0.val[3] = vld1q_f32(a_ptr + 16 * k_block + 12);

            // r0: fp32 to fp16
            float16x8x2_t r0_f16;
            float16x4_t r0_f16_0_h = vcvt_f16_f32(r0.val[0]);
            float16x4_t r0_f16_0_l = vcvt_f16_f32(r0.val[1]);
            float16x4_t r0_f16_1_h = vcvt_f16_f32(r0.val[2]);
            float16x4_t r0_f16_1_l = vcvt_f16_f32(r0.val[3]);
            r0_f16.val[0] = vcombine_f16(r0_f16_0_h, r0_f16_0_l);
            r0_f16.val[1] = vcombine_f16(r0_f16_1_h, r0_f16_1_l);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + 16 * k_block);
            int8x8_t n0_l = vget_low_s8(n0);
            int8x8_t n0_h = vget_high_s8(n0);
            int16x8_t n0_l_s16 = vmovl_s8(n0_l);
            int16x8_t n0_h_s16 = vmovl_s8(n0_h);
            float16x8_t n0_l_f16 = vcvtq_f16_s16(n0_l_s16);
            float16x8_t n0_h_f16 = vcvtq_f16_s16(n0_h_s16);
            acc_line_0.val[0] = vfmlalq_high_f16(acc_line_0.val[0], n0_h_f16, r0_f16.val[1]);
            acc_line_0.val[0] = vfmlalq_low_f16(acc_line_0.val[0], n0_h_f16, r0_f16.val[1]);
            acc_line_0.val[1] = vfmlalq_high_f16(acc_line_0.val[1], n0_l_f16, r0_f16.val[0]);
            acc_line_0.val[1] = vfmlalq_low_f16(acc_line_0.val[1], n0_l_f16, r0_f16.val[0]);

            // load from block, n = 1
            int8x16_t n1 = vld1q_s8(b_ptr + K + 16 * k_block);
            int8x8_t n1_l = vget_low_s8(n1);
            int8x8_t n1_h = vget_high_s8(n1);
            int16x8_t n1_l_s16 = vmovl_s8(n1_l);
            int16x8_t n1_h_s16 = vmovl_s8(n1_h);
            float16x8_t n1_l_f16 = vcvtq_f16_s16(n1_l_s16);
            float16x8_t n1_h_f16 = vcvtq_f16_s16(n1_h_s16);
            acc_line_1.val[0] = vfmlalq_high_f16(acc_line_1.val[0], n1_h_f16, r0_f16.val[1]);
            acc_line_1.val[0] = vfmlalq_low_f16(acc_line_1.val[0], n1_h_f16, r0_f16.val[1]);
            acc_line_1.val[1] = vfmlalq_high_f16(acc_line_1.val[1], n1_l_f16, r0_f16.val[0]);
            acc_line_1.val[1] = vfmlalq_low_f16(acc_line_1.val[1], n1_l_f16, r0_f16.val[0]);

            // load from block, n = 2
            int8x16_t n2 = vld1q_s8(b_ptr + 2 * K + 16 * k_block);
            int8x8_t n2_l = vget_low_s8(n2);
            int8x8_t n2_h = vget_high_s8(n2);
            int16x8_t n2_l_s16 = vmovl_s8(n2_l);
            int16x8_t n2_h_s16 = vmovl_s8(n2_h);
            float16x8_t n2_l_f16 = vcvtq_f16_s16(n2_l_s16);
            float16x8_t n2_h_f16 = vcvtq_f16_s16(n2_h_s16);
            acc_line_2.val[0] = vfmlalq_high_f16(acc_line_2.val[0], n2_h_f16, r0_f16.val[1]);
            acc_line_2.val[0] = vfmlalq_low_f16(acc_line_2.val[0], n2_h_f16, r0_f16.val[1]);
            acc_line_2.val[1] = vfmlalq_high_f16(acc_line_2.val[1], n2_l_f16, r0_f16.val[0]);
            acc_line_2.val[1] = vfmlalq_low_f16(acc_line_2.val[1], n2_l_f16, r0_f16.val[0]);

            // load from block, n = 3
            int8x16_t n3 = vld1q_s8(b_ptr + 3 * K + 16 * k_block);
            int8x8_t n3_l = vget_low_s8(n3);
            int8x8_t n3_h = vget_high_s8(n3);
            int16x8_t n3_l_s16 = vmovl_s8(n3_l);
            int16x8_t n3_h_s16 = vmovl_s8(n3_h);
            float16x8_t n3_l_f16 = vcvtq_f16_s16(n3_l_s16);
            float16x8_t n3_h_f16 = vcvtq_f16_s16(n3_h_s16);
            acc_line_3.val[0] = vfmlalq_high_f16(acc_line_3.val[0], n3_h_f16, r0_f16.val[1]);
            acc_line_3.val[0] = vfmlalq_low_f16(acc_line_3.val[0], n3_h_f16, r0_f16.val[1]);
            acc_line_3.val[1] = vfmlalq_high_f16(acc_line_3.val[1], n3_l_f16, r0_f16.val[0]);
            acc_line_3.val[1] = vfmlalq_low_f16(acc_line_3.val[1], n3_l_f16, r0_f16.val[0]);
        }

        // accumulate all
        float32x4_t acc_line_0_sum = vaddq_f32(acc_line_0.val[0], acc_line_0.val[1]);
        float32_t acc_line_0_sum_f32 = vaddvq_f32(acc_line_0_sum);
        *(c_ptr + 4 * n_block + 0) = acc_line_0_sum_f32 * scale;

        float32x4_t acc_line_1_sum = vaddq_f32(acc_line_1.val[0], acc_line_1.val[1]);
        float32_t acc_line_1_sum_f32 = vaddvq_f32(acc_line_1_sum);
        *(c_ptr + 4 * n_block + 1) = acc_line_1_sum_f32 * scale;

        float32x4_t acc_line_2_sum = vaddq_f32(acc_line_2.val[0], acc_line_2.val[1]);
        float32_t acc_line_2_sum_f32 = vaddvq_f32(acc_line_2_sum);
        *(c_ptr + 4 * n_block + 2) = acc_line_2_sum_f32 * scale;

        float32x4_t acc_line_3_sum = vaddq_f32(acc_line_3.val[0], acc_line_3.val[1]);
        float32_t acc_line_3_sum_f32 = vaddvq_f32(acc_line_3_sum);
        *(c_ptr + 4 * n_block + 3) = acc_line_3_sum_f32 * scale;
    }

    // the left.
    for (int32_t n = 0; n < n_blocks_left; n++) {
        auto a_ptr = (float *)A;
        auto b_ptr = (int8_t *)B + 4 * (N / 4) * K;
        auto c_ptr = (float *)C + 4 * (N / 4);

        float32x4x2_t acc_line_0;
        acc_line_0.val[0] = vdupq_n_f32(0.f);
        acc_line_0.val[1] = vdupq_n_f32(0.f);

        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            float32x4x4_t r0;
            r0.val[0] = vld1q_f32(a_ptr + 16 * k_block);
            r0.val[1] = vld1q_f32(a_ptr + 16 * k_block + 4);
            r0.val[2] = vld1q_f32(a_ptr + 16 * k_block + 8);
            r0.val[3] = vld1q_f32(a_ptr + 16 * k_block + 12);

            // r0: fp32 to fp16
            float16x8x2_t r0_f16;
            float16x4_t r0_f16_0_h = vcvt_f16_f32(r0.val[0]);
            float16x4_t r0_f16_0_l = vcvt_f16_f32(r0.val[1]);
            float16x4_t r0_f16_1_h = vcvt_f16_f32(r0.val[2]);
            float16x4_t r0_f16_1_l = vcvt_f16_f32(r0.val[3]);
            r0_f16.val[0] = vcombine_f16(r0_f16_0_h, r0_f16_0_l);
            r0_f16.val[1] = vcombine_f16(r0_f16_1_h, r0_f16_1_l);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + n * K + 16 * k_block);
            int8x8_t n0_l = vget_low_s8(n0);
            int8x8_t n0_h = vget_high_s8(n0);
            int16x8_t n0_l_s16 = vmovl_s8(n0_l);
            int16x8_t n0_h_s16 = vmovl_s8(n0_h);
            float16x8_t n0_l_f16 = vcvtq_f16_s16(n0_l_s16);
            float16x8_t n0_h_f16 = vcvtq_f16_s16(n0_h_s16);
            acc_line_0.val[0] = vfmlalq_high_f16(acc_line_0.val[0], n0_h_f16, r0_f16.val[1]);
            acc_line_0.val[0] = vfmlalq_low_f16(acc_line_0.val[0], n0_h_f16, r0_f16.val[1]);
            acc_line_0.val[1] = vfmlalq_high_f16(acc_line_0.val[1], n0_l_f16, r0_f16.val[0]);
            acc_line_0.val[1] = vfmlalq_low_f16(acc_line_0.val[1], n0_l_f16, r0_f16.val[0]);
        }

        // accumulate
        // accumulate all
        float32x4_t acc_line_0_sum = vaddq_f32(acc_line_0.val[0], acc_line_0.val[1]);
        float32_t acc_line_0_sum_f32 = vaddvq_f32(acc_line_0_sum);
        *(c_ptr + n) = acc_line_0_sum_f32 * scale;
    }
}

void fp32_qt8_fp32_kernel_4x4(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
                              bool transpose_b) {
    if (!transpose_b) {
        // Not Supported Yet.
        return;
    }

    // I tile K with 16 element per block. the 16 element will be load to 128bit vector.
    if (K % 16) abort();

    int32_t k_blocks = K / 16;
    float scale = SA * SB;

    auto a_ptr = (float *)A;
    auto b_ptr = (int8_t *)B;
    auto c_ptr = (float *)C;

    float32x4x4_t acc_line_0_0 = {vmovq_n_f32(0.f), vmovq_n_f32(0.f), vmovq_n_f32(0.f),
                                  vmovq_n_f32(0.f)}; // line1 ele 1 and 2
    float32x4x4_t acc_line_0_1 = {vmovq_n_f32(0.f), vmovq_n_f32(0.f), vmovq_n_f32(0.f),
                                  vmovq_n_f32(0.f)}; // line1 ele 3 and 4
    float32x4x4_t acc_line_1_0 = {vmovq_n_f32(0.f), vmovq_n_f32(0.f), vmovq_n_f32(0.f),
                                  vmovq_n_f32(0.f)}; // line 2 ele 1 and 2
    float32x4x4_t acc_line_1_1 = {vmovq_n_f32(0.f), vmovq_n_f32(0.f), vmovq_n_f32(0.f),
                                  vmovq_n_f32(0.f)}; // line 2 ele 3 and 4
    float32x4x4_t acc_line_2_0 = {vmovq_n_f32(0.f), vmovq_n_f32(0.f), vmovq_n_f32(0.f),
                                  vmovq_n_f32(0.f)}; // line 3 ele 1 and 2
    float32x4x4_t acc_line_2_1 = {vmovq_n_f32(0.f), vmovq_n_f32(0.f), vmovq_n_f32(0.f),
                                  vmovq_n_f32(0.f)}; // line 3 ele 3 and 4
    float32x4x4_t acc_line_3_0 = {vmovq_n_f32(0.f), vmovq_n_f32(0.f), vmovq_n_f32(0.f),
                                  vmovq_n_f32(0.f)}; // line 4 ele 1 and 2
    float32x4x4_t acc_line_3_1 = {vmovq_n_f32(0.f), vmovq_n_f32(0.f), vmovq_n_f32(0.f),
                                  vmovq_n_f32(0.f)}; // line 4 ele 3 and 4

    float acc_0[4] = {0.f, 0.f, 0.f, 0.f};
    float acc_1[4] = {0.f, 0.f, 0.f, 0.f};
    float acc_2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc_3[4] = {0.f, 0.f, 0.f, 0.f};

    for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
        // load four lines from a
        // a: line 0
        float32x4x4_t a0;
        a0.val[0] = vld1q_f32(a_ptr + 16 * k_block);
        a0.val[1] = vld1q_f32(a_ptr + 16 * k_block + 4);
        a0.val[2] = vld1q_f32(a_ptr + 16 * k_block + 8);
        a0.val[3] = vld1q_f32(a_ptr + 16 * k_block + 12);
        // a0: fp32 to fp16
        float16x8x2_t a0_f16;
        float16x4_t a0_f16_0_h = vcvt_f16_f32(a0.val[0]);
        float16x4_t a0_f16_0_l = vcvt_f16_f32(a0.val[1]);
        float16x4_t a0_f16_1_h = vcvt_f16_f32(a0.val[2]);
        float16x4_t a0_f16_1_l = vcvt_f16_f32(a0.val[3]);
        a0_f16.val[0] = vcombine_f16(a0_f16_0_h, a0_f16_0_l);
        a0_f16.val[1] = vcombine_f16(a0_f16_1_h, a0_f16_1_l);

        // a: line 1
        float32x4x4_t a1;
        a1.val[0] = vld1q_f32(a_ptr + K + 16 * k_block);
        a1.val[1] = vld1q_f32(a_ptr + K + 16 * k_block + 4);
        a1.val[2] = vld1q_f32(a_ptr + K + 16 * k_block + 8);
        a1.val[3] = vld1q_f32(a_ptr + K + 16 * k_block + 12);
        // a1: fp32 to fp16
        float16x8x2_t a1_f16;
        float16x4_t a1_f16_0_h = vcvt_f16_f32(a1.val[0]);
        float16x4_t a1_f16_0_l = vcvt_f16_f32(a1.val[1]);
        float16x4_t a1_f16_1_h = vcvt_f16_f32(a1.val[2]);
        float16x4_t a1_f16_1_l = vcvt_f16_f32(a1.val[3]);
        a1_f16.val[0] = vcombine_f16(a1_f16_0_h, a1_f16_0_l);
        a1_f16.val[1] = vcombine_f16(a1_f16_1_h, a1_f16_1_l);

        // a: line 2
        float32x4x4_t a2;
        a2.val[0] = vld1q_f32(a_ptr + 2 * K + 16 * k_block);
        a2.val[1] = vld1q_f32(a_ptr + 2 * K + 16 * k_block + 4);
        a2.val[2] = vld1q_f32(a_ptr + 2 * K + 16 * k_block + 8);
        a2.val[3] = vld1q_f32(a_ptr + 2 * K + 16 * k_block + 12);
        // a2: fp32 to fp16
        float16x8x2_t a2_f16;
        float16x4_t a2_f16_0_h = vcvt_f16_f32(a2.val[0]);
        float16x4_t a2_f16_0_l = vcvt_f16_f32(a2.val[1]);
        float16x4_t a2_f16_1_h = vcvt_f16_f32(a2.val[2]);
        float16x4_t a2_f16_1_l = vcvt_f16_f32(a2.val[3]);
        a2_f16.val[0] = vcombine_f16(a2_f16_0_h, a2_f16_0_l);
        a2_f16.val[1] = vcombine_f16(a2_f16_1_h, a2_f16_1_l);

        // a: line 3
        float32x4x4_t a3;
        a3.val[0] = vld1q_f32(a_ptr + 3 * K + 16 * k_block);
        a3.val[1] = vld1q_f32(a_ptr + 3 * K + 16 * k_block + 4);
        a3.val[2] = vld1q_f32(a_ptr + 3 * K + 16 * k_block + 8);
        a3.val[3] = vld1q_f32(a_ptr + 3 * K + 16 * k_block + 12);
        // a3: fp32 to fp16
        float16x8x2_t a3_f16;
        float16x4_t a3_f16_0_h = vcvt_f16_f32(a3.val[0]);
        float16x4_t a3_f16_0_l = vcvt_f16_f32(a3.val[1]);
        float16x4_t a3_f16_1_h = vcvt_f16_f32(a3.val[2]);
        float16x4_t a3_f16_1_l = vcvt_f16_f32(a3.val[3]);
        a3_f16.val[0] = vcombine_f16(a3_f16_0_h, a3_f16_0_l);
        a3_f16.val[1] = vcombine_f16(a3_f16_1_h, a3_f16_1_l);

        // load four lines from b
        // b: line 0
        int8x16_t n0 = vld1q_s8(b_ptr + 16 * k_block);
        int8x8_t n0_l = vget_low_s8(n0);
        int8x8_t n0_h = vget_high_s8(n0);
        int16x8_t n0_l_s16 = vmovl_s8(n0_l);
        int16x8_t n0_h_s16 = vmovl_s8(n0_h);
        float16x8_t n0_l_f16 = vcvtq_f16_s16(n0_l_s16);
        float16x8_t n0_h_f16 = vcvtq_f16_s16(n0_h_s16);

        // b: line 1
        int8x16_t n1 = vld1q_s8(b_ptr + K + 16 * k_block);
        int8x8_t n1_l = vget_low_s8(n1);
        int8x8_t n1_h = vget_high_s8(n1);
        int16x8_t n1_l_s16 = vmovl_s8(n1_l);
        int16x8_t n1_h_s16 = vmovl_s8(n1_h);
        float16x8_t n1_l_f16 = vcvtq_f16_s16(n1_l_s16);
        float16x8_t n1_h_f16 = vcvtq_f16_s16(n1_h_s16);

        // b: line 2
        int8x16_t n2 = vld1q_s8(b_ptr + 2 * K + 16 * k_block);
        int8x8_t n2_l = vget_low_s8(n2);
        int8x8_t n2_h = vget_high_s8(n2);
        int16x8_t n2_l_s16 = vmovl_s8(n2_l);
        int16x8_t n2_h_s16 = vmovl_s8(n2_h);
        float16x8_t n2_l_f16 = vcvtq_f16_s16(n2_l_s16);
        float16x8_t n2_h_f16 = vcvtq_f16_s16(n2_h_s16);

        // b: line 3
        int8x16_t n3 = vld1q_s8(b_ptr + 3 * K + 16 * k_block);
        int8x8_t n3_l = vget_low_s8(n3);
        int8x8_t n3_h = vget_high_s8(n3);
        int16x8_t n3_l_s16 = vmovl_s8(n3_l);
        int16x8_t n3_h_s16 = vmovl_s8(n3_h);
        float16x8_t n3_l_f16 = vcvtq_f16_s16(n3_l_s16);
        float16x8_t n3_h_f16 = vcvtq_f16_s16(n3_h_s16);

        // accumulate
        // (0, 0)
        acc_line_0_0.val[0] = vfmlalq_high_f16(acc_line_0_0.val[0], n0_h_f16, a0_f16.val[1]);
        acc_line_0_0.val[0] = vfmlalq_low_f16(acc_line_0_0.val[0], n0_h_f16, a0_f16.val[1]);
        acc_line_0_0.val[1] = vfmlalq_high_f16(acc_line_0_0.val[1], n0_l_f16, a0_f16.val[0]);
        acc_line_0_0.val[1] = vfmlalq_low_f16(acc_line_0_0.val[1], n0_l_f16, a0_f16.val[0]);

        // (0, 1)
        acc_line_0_0.val[2] = vfmlalq_high_f16(acc_line_0_0.val[2], n1_h_f16, a0_f16.val[1]);
        acc_line_0_0.val[2] = vfmlalq_low_f16(acc_line_0_0.val[2], n1_h_f16, a0_f16.val[1]);
        acc_line_0_0.val[3] = vfmlalq_high_f16(acc_line_0_0.val[3], n1_l_f16, a0_f16.val[0]);
        acc_line_0_0.val[3] = vfmlalq_low_f16(acc_line_0_0.val[3], n1_l_f16, a0_f16.val[0]);

        // (0, 2)
        acc_line_0_1.val[0] = vfmlalq_high_f16(acc_line_0_1.val[0], n2_h_f16, a0_f16.val[1]);
        acc_line_0_1.val[0] = vfmlalq_low_f16(acc_line_0_1.val[0], n2_h_f16, a0_f16.val[1]);
        acc_line_0_1.val[1] = vfmlalq_high_f16(acc_line_0_1.val[1], n2_l_f16, a0_f16.val[0]);
        acc_line_0_1.val[1] = vfmlalq_low_f16(acc_line_0_1.val[1], n2_l_f16, a0_f16.val[0]);

        // (0, 3)
        acc_line_0_1.val[2] = vfmlalq_high_f16(acc_line_0_1.val[2], n3_h_f16, a0_f16.val[1]);
        acc_line_0_1.val[2] = vfmlalq_low_f16(acc_line_0_1.val[2], n3_h_f16, a0_f16.val[1]);
        acc_line_0_1.val[3] = vfmlalq_high_f16(acc_line_0_1.val[3], n3_l_f16, a0_f16.val[0]);
        acc_line_0_1.val[3] = vfmlalq_low_f16(acc_line_0_1.val[3], n3_l_f16, a0_f16.val[0]);

        // (1, 0)
        acc_line_1_0.val[0] = vfmlalq_high_f16(acc_line_1_0.val[0], n0_h_f16, a1_f16.val[1]);
        acc_line_1_0.val[0] = vfmlalq_low_f16(acc_line_1_0.val[0], n0_h_f16, a1_f16.val[1]);
        acc_line_1_0.val[1] = vfmlalq_high_f16(acc_line_1_0.val[1], n0_l_f16, a1_f16.val[0]);
        acc_line_1_0.val[1] = vfmlalq_low_f16(acc_line_1_0.val[1], n0_l_f16, a1_f16.val[0]);

        // (1, 1)
        acc_line_1_0.val[2] = vfmlalq_high_f16(acc_line_1_0.val[2], n1_h_f16, a1_f16.val[1]);
        acc_line_1_0.val[2] = vfmlalq_low_f16(acc_line_1_0.val[2], n1_h_f16, a1_f16.val[1]);
        acc_line_1_0.val[3] = vfmlalq_high_f16(acc_line_1_0.val[3], n1_l_f16, a1_f16.val[0]);
        acc_line_1_0.val[3] = vfmlalq_low_f16(acc_line_1_0.val[3], n1_l_f16, a1_f16.val[0]);

        // (1, 2)
        acc_line_1_1.val[0] = vfmlalq_high_f16(acc_line_1_1.val[0], n2_h_f16, a1_f16.val[1]);
        acc_line_1_1.val[0] = vfmlalq_low_f16(acc_line_1_1.val[0], n2_h_f16, a1_f16.val[1]);
        acc_line_1_1.val[1] = vfmlalq_high_f16(acc_line_1_1.val[1], n2_l_f16, a1_f16.val[0]);
        acc_line_1_1.val[1] = vfmlalq_low_f16(acc_line_1_1.val[1], n2_l_f16, a1_f16.val[0]);

        // (1, 3)
        acc_line_1_1.val[2] = vfmlalq_high_f16(acc_line_1_1.val[2], n3_h_f16, a1_f16.val[1]);
        acc_line_1_1.val[2] = vfmlalq_low_f16(acc_line_1_1.val[2], n3_h_f16, a1_f16.val[1]);
        acc_line_1_1.val[3] = vfmlalq_high_f16(acc_line_1_1.val[3], n3_l_f16, a1_f16.val[0]);
        acc_line_1_1.val[3] = vfmlalq_low_f16(acc_line_1_1.val[3], n3_l_f16, a1_f16.val[0]);

        // (2, 0)
        acc_line_2_0.val[0] = vfmlalq_high_f16(acc_line_2_0.val[0], n0_h_f16, a2_f16.val[1]);
        acc_line_2_0.val[0] = vfmlalq_low_f16(acc_line_2_0.val[0], n0_h_f16, a2_f16.val[1]);
        acc_line_2_0.val[1] = vfmlalq_high_f16(acc_line_2_0.val[1], n0_l_f16, a2_f16.val[0]);
        acc_line_2_0.val[1] = vfmlalq_low_f16(acc_line_2_0.val[1], n0_l_f16, a2_f16.val[0]);

        // (2, 1)
        acc_line_2_0.val[2] = vfmlalq_high_f16(acc_line_2_0.val[2], n1_h_f16, a2_f16.val[1]);
        acc_line_2_0.val[2] = vfmlalq_low_f16(acc_line_2_0.val[2], n1_h_f16, a2_f16.val[1]);
        acc_line_2_0.val[3] = vfmlalq_high_f16(acc_line_2_0.val[3], n1_l_f16, a2_f16.val[0]);
        acc_line_2_0.val[3] = vfmlalq_low_f16(acc_line_2_0.val[3], n1_l_f16, a2_f16.val[0]);

        // (2, 2)
        acc_line_2_1.val[0] = vfmlalq_high_f16(acc_line_2_1.val[0], n2_h_f16, a2_f16.val[1]);
        acc_line_2_1.val[0] = vfmlalq_low_f16(acc_line_2_1.val[0], n2_h_f16, a2_f16.val[1]);
        acc_line_2_1.val[1] = vfmlalq_high_f16(acc_line_2_1.val[1], n2_l_f16, a2_f16.val[0]);
        acc_line_2_1.val[1] = vfmlalq_low_f16(acc_line_2_1.val[1], n2_l_f16, a2_f16.val[0]);

        // (2, 3)
        acc_line_2_1.val[2] = vfmlalq_high_f16(acc_line_2_1.val[2], n3_h_f16, a2_f16.val[1]);
        acc_line_2_1.val[2] = vfmlalq_low_f16(acc_line_2_1.val[2], n3_h_f16, a2_f16.val[1]);
        acc_line_2_1.val[3] = vfmlalq_high_f16(acc_line_2_1.val[3], n3_l_f16, a2_f16.val[0]);
        acc_line_2_1.val[3] = vfmlalq_low_f16(acc_line_2_1.val[3], n3_l_f16, a2_f16.val[0]);

        // (3, 0)
        acc_line_3_0.val[0] = vfmlalq_high_f16(acc_line_3_0.val[0], n0_h_f16, a3_f16.val[1]);
        acc_line_3_0.val[0] = vfmlalq_low_f16(acc_line_3_0.val[0], n0_h_f16, a3_f16.val[1]);
        acc_line_3_0.val[1] = vfmlalq_high_f16(acc_line_3_0.val[1], n0_l_f16, a3_f16.val[0]);
        acc_line_3_0.val[1] = vfmlalq_low_f16(acc_line_3_0.val[1], n0_l_f16, a3_f16.val[0]);

        // (3, 1)
        acc_line_3_0.val[2] = vfmlalq_high_f16(acc_line_3_0.val[2], n1_h_f16, a3_f16.val[1]);
        acc_line_3_0.val[2] = vfmlalq_low_f16(acc_line_3_0.val[2], n1_h_f16, a3_f16.val[1]);
        acc_line_3_0.val[3] = vfmlalq_high_f16(acc_line_3_0.val[3], n1_l_f16, a3_f16.val[0]);
        acc_line_3_0.val[3] = vfmlalq_low_f16(acc_line_3_0.val[3], n1_l_f16, a3_f16.val[0]);

        // (3, 2)
        acc_line_3_1.val[0] = vfmlalq_high_f16(acc_line_3_1.val[0], n2_h_f16, a3_f16.val[1]);
        acc_line_3_1.val[0] = vfmlalq_low_f16(acc_line_3_1.val[0], n2_h_f16, a3_f16.val[1]);
        acc_line_3_1.val[1] = vfmlalq_high_f16(acc_line_3_1.val[1], n2_l_f16, a3_f16.val[0]);
        acc_line_3_1.val[1] = vfmlalq_low_f16(acc_line_3_1.val[1], n2_l_f16, a3_f16.val[0]);

        // (3, 3)
        acc_line_3_1.val[2] = vfmlalq_high_f16(acc_line_3_1.val[2], n3_h_f16, a3_f16.val[1]);
        acc_line_3_1.val[2] = vfmlalq_low_f16(acc_line_3_1.val[2], n3_h_f16, a3_f16.val[1]);
        acc_line_3_1.val[3] = vfmlalq_high_f16(acc_line_3_1.val[3], n3_l_f16, a3_f16.val[0]);
        acc_line_3_1.val[3] = vfmlalq_low_f16(acc_line_3_1.val[3], n3_l_f16, a3_f16.val[0]);
    }

    // store
    // (0, 0)
    float32x4_t acc_line_0_0_sum = vaddq_f32(acc_line_0_0.val[0], acc_line_0_0.val[1]);
    float32_t acc_line_0_0_sum_f32 = vaddvq_f32(acc_line_0_0_sum);
    acc_0[0] = acc_line_0_0_sum_f32;

    // (0, 1)
    float32x4_t acc_line_0_1_sum = vaddq_f32(acc_line_0_0.val[2], acc_line_0_0.val[3]);
    float32_t acc_line_0_1_sum_f32 = vaddvq_f32(acc_line_0_1_sum);
    acc_0[1] = acc_line_0_1_sum_f32;

    // (0, 2)
    float32x4_t acc_line_0_2_sum = vaddq_f32(acc_line_0_1.val[0], acc_line_0_1.val[1]);
    float32_t acc_line_0_2_sum_f32 = vaddvq_f32(acc_line_0_2_sum);
    acc_0[2] = acc_line_0_2_sum_f32;

    // (0, 3)
    float32x4_t acc_line_0_3_sum = vaddq_f32(acc_line_0_1.val[2], acc_line_0_1.val[3]);
    float32_t acc_line_0_3_sum_f32 = vaddvq_f32(acc_line_0_3_sum);
    acc_0[3] = acc_line_0_3_sum_f32;

    // (1, 0)
    float32x4_t acc_line_1_0_sum = vaddq_f32(acc_line_1_0.val[0], acc_line_1_0.val[1]);
    float32_t acc_line_1_0_sum_f32 = vaddvq_f32(acc_line_1_0_sum);
    acc_1[0] = acc_line_1_0_sum_f32;

    // (1, 1)
    float32x4_t acc_line_1_1_sum = vaddq_f32(acc_line_1_0.val[2], acc_line_1_0.val[3]);
    float32_t acc_line_1_1_sum_f32 = vaddvq_f32(acc_line_1_1_sum);
    acc_1[1] = acc_line_1_1_sum_f32;

    // (1, 2)
    float32x4_t acc_line_1_2_sum = vaddq_f32(acc_line_1_1.val[0], acc_line_1_1.val[1]);
    float32_t acc_line_1_2_sum_f32 = vaddvq_f32(acc_line_1_2_sum);
    acc_1[2] = acc_line_1_2_sum_f32;

    // (1, 3)
    float32x4_t acc_line_1_3_sum = vaddq_f32(acc_line_1_1.val[2], acc_line_1_1.val[3]);
    float32_t acc_line_1_3_sum_f32 = vaddvq_f32(acc_line_1_3_sum);
    acc_1[3] = acc_line_1_3_sum_f32;

    // (2, 0)
    float32x4_t acc_line_2_0_sum = vaddq_f32(acc_line_2_0.val[0], acc_line_2_0.val[1]);
    float32_t acc_line_2_0_sum_f32 = vaddvq_f32(acc_line_2_0_sum);
    acc_2[0] = acc_line_2_0_sum_f32;

    // (2, 1)
    float32x4_t acc_line_2_1_sum = vaddq_f32(acc_line_2_0.val[2], acc_line_2_0.val[3]);
    float32_t acc_line_2_1_sum_f32 = vaddvq_f32(acc_line_2_1_sum);
    acc_2[1] = acc_line_2_1_sum_f32;

    // (2, 2)
    float32x4_t acc_line_2_2_sum = vaddq_f32(acc_line_2_1.val[0], acc_line_2_1.val[1]);
    float32_t acc_line_2_2_sum_f32 = vaddvq_f32(acc_line_2_2_sum);
    acc_2[2] = acc_line_2_2_sum_f32;

    // (2, 3)
    float32x4_t acc_line_2_3_sum = vaddq_f32(acc_line_2_1.val[2], acc_line_2_1.val[3]);
    float32_t acc_line_2_3_sum_f32 = vaddvq_f32(acc_line_2_3_sum);
    acc_2[3] = acc_line_2_3_sum_f32;

    // (3, 0)
    float32x4_t acc_line_3_0_sum = vaddq_f32(acc_line_3_0.val[0], acc_line_3_0.val[1]);
    float32_t acc_line_3_0_sum_f32 = vaddvq_f32(acc_line_3_0_sum);
    acc_3[0] = acc_line_3_0_sum_f32;

    // (3, 1)
    float32x4_t acc_line_3_1_sum = vaddq_f32(acc_line_3_0.val[2], acc_line_3_0.val[3]);
    float32_t acc_line_3_1_sum_f32 = vaddvq_f32(acc_line_3_1_sum);
    acc_3[1] = acc_line_3_1_sum_f32;

    // (3, 2)
    float32x4_t acc_line_3_2_sum = vaddq_f32(acc_line_3_1.val[0], acc_line_3_1.val[1]);
    float32_t acc_line_3_2_sum_f32 = vaddvq_f32(acc_line_3_2_sum);
    acc_3[2] = acc_line_3_2_sum_f32;

    // (3, 3)
    float32x4_t acc_line_3_3_sum = vaddq_f32(acc_line_3_1.val[2], acc_line_3_1.val[3]);
    float32_t acc_line_3_3_sum_f32 = vaddvq_f32(acc_line_3_3_sum);
    acc_3[3] = acc_line_3_3_sum_f32;

    float32x4_t acc_final_0 = vld1q_f32(acc_0);
    float32x4_t acc_final_1 = vld1q_f32(acc_1);
    float32x4_t acc_final_2 = vld1q_f32(acc_2);
    float32x4_t acc_final_3 = vld1q_f32(acc_3);

    float32x4_t acc_scaled_0 = vmulq_n_f32(acc_final_0, scale);
    float32x4_t acc_scaled_1 = vmulq_n_f32(acc_final_1, scale);
    float32x4_t acc_scaled_2 = vmulq_n_f32(acc_final_2, scale);
    float32x4_t acc_scaled_3 = vmulq_n_f32(acc_final_3, scale);

    vst1q_f32(c_ptr, acc_scaled_0);
    vst1q_f32(c_ptr + N, acc_scaled_1);
    vst1q_f32(c_ptr + 2 * N, acc_scaled_2);
    vst1q_f32(c_ptr + 3 * N, acc_scaled_3);
}

void fp32_qt8_fp32_gemm(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                        float SB, bool transpose_b) {
    if (M == 1) {
        fp32_qt8_fp32_gemv(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        auto a_ptr = (float *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;

        int32_t m_blocks = M / 4;
        int32_t n_blocks = N / 4;
        int32_t m_left = M % 4;
        int32_t n_left = N % 4;

        if (M < 4 || N < 4) {
            for (int m = 0; m < M; ++m) {
                fp32_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
            return;
        }

        // main loop
        for (int m = 0; m < m_blocks; ++m) {
            for (int n = 0; n < n_blocks; ++n) {
                fp32_qt8_fp32_kernel_4x4(a_ptr + 4 * m * K, b_ptr + 4 * n * K, c_ptr + 4 * m * N + 4 * n, N,
                                         K, SA, SB, transpose_b);
            }

            // some re cauculating may be needed
            if (n_left) {
                fp32_qt8_fp32_kernel_4x4(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                         c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA, SB,
                                         transpose_b);
            }
        }

        if (m_left) {
            for (int m = m_blocks * 4; m < M; ++m) {
                fp32_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
        }
    }
}

void fp32_qt8_fp32_gemm_omp(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                            float SB, bool transpose_b) {
    if (M == 1) {
        fp32_qt8_fp32_gemv(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        auto a_ptr = (float *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;

        int32_t m_blocks = M / 4;
        int32_t n_blocks = N / 4;
        int32_t m_left = M % 4;
        int32_t n_left = N % 4;

        if (M < 4 || N < 4) {
            for (int m = 0; m < M; ++m) {
                fp32_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
            return;
        }

        if (M > 64) {
            // main loop
#pragma omp parallel for num_threads(4)
            for (int m = 0; m < m_blocks; ++m) {
                for (int n = 0; n < n_blocks; ++n) {
                    fp32_qt8_fp32_kernel_4x4(a_ptr + 4 * m * K, b_ptr + 4 * n * K, c_ptr + 4 * m * N + 4 * n,
                                             N, K, SA, SB, transpose_b);
                }

                // some re cauculating may be needed
                if (n_left) {
                    fp32_qt8_fp32_kernel_4x4(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                             c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA, SB,
                                             transpose_b);
                }
            }

            if (m_left) {
                for (int m = m_blocks * 4; m < M; ++m) {
                    fp32_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
                }
            }
        } else {
            // main loop
            for (int m = 0; m < m_blocks; ++m) {
                for (int n = 0; n < n_blocks; ++n) {
                    fp32_qt8_fp32_kernel_4x4(a_ptr + 4 * m * K, b_ptr + 4 * n * K, c_ptr + 4 * m * N + 4 * n,
                                             N, K, SA, SB, transpose_b);
                }

                // some re cauculating may be needed
                if (n_left) {
                    fp32_qt8_fp32_kernel_4x4(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                             c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA, SB,
                                             transpose_b);
                }
            }

            if (m_left) {
                for (int m = m_blocks * 4; m < M; ++m) {
                    fp32_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
                }
            }
        }
    }
}

} // namespace mllm::armv8

#endif