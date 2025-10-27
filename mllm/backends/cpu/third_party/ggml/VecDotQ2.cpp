/*
 * This code is based on ggml(https://github.com/ggerganov/ggml),
 * please see https://github.com/ggerganov/ggml/blob/master/src/ggml.c
 * ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
 *
 * MIT License
 * Copyright (c) 2022 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "VecDotQ2.hpp"
#include "ComputeUtils.hpp"

void vec_dot_q2_0_q8_0(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);

    const auto *__restrict x = static_cast<const block_q2_0 *>(vx);
    const auto *__restrict y = static_cast<const block_q8_0 *>(vy);

#if defined(__AVX2__)
    // AVX2 implementation
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const __m256 d_vec = _mm256_set1_ps(MLLM_FP16_TO_FP32(x[i].d) * MLLM_FP16_TO_FP32(y[i].d));

        const __m128i q2_packed = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(x[i].qs));

        const __m256i pshufb_mask = _mm256_setr_epi8(
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
        const __m256i q2_bytes = _mm256_shuffle_epi8(_mm256_set_m128i(_mm_setzero_si128(), q2_packed), pshufb_mask);

        const __m256i shift_const = _mm256_set_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        const __m256i q2_shifted = _mm256_srlv_epi32(q2_bytes, shift_const);
        const __m256i q2_isolated = _mm256_and_si256(q2_shifted, _mm256_set1_epi8(0x03));
        const __m256i q2_final = _mm256_sub_epi8(q2_isolated, _mm256_set1_epi8(2));

        const __m256i q8_data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[i].qs));

        const __m256i products = _mm256_maddubs_epi16(q2_final, q8_data);
        const __m256i sum_lanes = _mm256_madd_epi16(_mm256_set1_epi16(1), products);

        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum_lanes), d_vec, acc);
    }
    *s = hsum_float_8(acc);

#elif defined(__ARM_NEON)
    // ARM NEON implementation
    float32x4_t sumv = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i++) {
        const float d = MLLM_FP16_TO_FP32(x[i].d) * MLLM_FP16_TO_FP32(y[i].d);
        const float32x4_t d_vec = vdupq_n_f32(d);

        int32x4_t isum_block = vdupq_n_s32(0);

        const uint8_t *q2_ptr = x[i].qs;
        const int8_t *q8_ptr = y[i].qs;

        // Unpack 8 bytes of Q2 data into a temporary 32-byte array
        int8_t q2_unpacked[32];
        for (int j = 0; j < 8; ++j) {
            uint8_t b = q2_ptr[j];
            q2_unpacked[j * 4 + 0] = ((b >> 0) & 3) - 2;
            q2_unpacked[j * 4 + 1] = ((b >> 2) & 3) - 2;
            q2_unpacked[j * 4 + 2] = ((b >> 4) & 3) - 2;
            q2_unpacked[j * 4 + 3] = ((b >> 6) & 3) - 2;
        }

        // Perform dot product on unpacked data
        const int8x16_t q2_v0 = vld1q_s8(&q2_unpacked[0]);
        const int8x16_t q2_v1 = vld1q_s8(&q2_unpacked[16]);

        const int8x16_t q8_v0 = vld1q_s8(q8_ptr);
        const int8x16_t q8_v1 = vld1q_s8(q8_ptr + 16);

        const int16x8_t p0 = vmull_s8(vget_low_s8(q2_v0), vget_low_s8(q8_v0));
        const int16x8_t p1 = vmull_s8(vget_high_s8(q2_v0), vget_high_s8(q8_v0));
        const int16x8_t p2 = vmull_s8(vget_low_s8(q2_v1), vget_low_s8(q8_v1));
        const int16x8_t p3 = vmull_s8(vget_high_s8(q2_v1), vget_high_s8(q8_v1));

        isum_block = vcombine_s32(
            vpadd_s32(vpaddl_s16(vget_low_s16(p0)), vpaddl_s16(vget_high_s16(p0))),
            vpadd_s32(vpaddl_s16(vget_low_s16(p1)), vpaddl_s16(vget_high_s16(p1))));
        isum_block = vaddq_s32(isum_block, vcombine_s32(
                                               vpadd_s32(vpaddl_s16(vget_low_s16(p2)), vpaddl_s16(vget_high_s16(p2))),
                                               vpadd_s32(vpaddl_s16(vget_low_s16(p3)), vpaddl_s16(vget_high_s16(p3)))));

        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(isum_block), d_vec);
    }
    *s = vaddvq_f32(sumv);

#else
    // Fallback scalar implementation
    float sumf = 0.0;
    for (int i = 0; i < nb; ++i) {
        const float d = MLLM_FP16_TO_FP32(x[i].d) * MLLM_FP16_TO_FP32(y[i].d);
        int32_t isum = 0;
        for (int j = 0; j < qk / 4; ++j) {
            uint8_t packed_q2 = x[i].qs[j];
            for (int l = 0; l < 4; ++l) {
                const int8_t x0 = ((packed_q2 >> (l * 2)) & 3) - 2;
                isum += x0 * y[i].qs[j * 4 + l];
            }
        }
        sumf += d * isum;
    }
    *s = sumf;
#endif
}

void vec_dot_q2_K_q8_K(int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    const block_q2_K *__restrict x = (block_q2_K *)vx;
    const block_q8_K *__restrict y = (block_q8_K *)vy;

    const int nb = n / QK_K;

#ifdef __ARM_FEATURE_SVE
    const int vector_length = svcntb() * 8;
    const svuint8_t m3s = svdup_n_u8(0x3);
    const svuint32_t m4s = svdup_n_u32(0xF);
    const svint32_t vzero_sv = svdup_n_s32(0);
    svfloat32_t acc_sum = svdup_n_f32(0);
    svbool_t pred_s32 = svptrue_pat_b32(SV_VL4);

    switch (vector_length) {
    case 128:
        for (int i = 0; i < nb; ++i) {
            const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
            svfloat32_t d_broad = svdup_n_f32((float32_t)d);
            const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);
            svfloat32_t dmin_broad = svdup_n_f32((float32_t)dmin);

            const uint8_t *__restrict q2 = x[i].qs;
            const int8_t *__restrict q8_sv = y[i].qs;
            const uint8_t *__restrict sc = x[i].scales;

            svuint32_t mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc);
            const svint32_t mins_sv_1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

            mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc + 4);
            const svint32_t mins_sv_2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

            svint32_t q8sums_sv_1 = svld1sh_s32(svptrue_b32(), y[i].bsums);
            svint32_t q8sums_sv_2 = svld1sh_s32(svptrue_b32(), y[i].bsums + 4);

            const svint32_t s0 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), mins_sv_1, q8sums_sv_1), svmul_s32_x(svptrue_b32(), mins_sv_2, q8sums_sv_2));

            mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc + 8);
            const svint32_t mins_sv_3 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

            mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc + 12);
            const svint32_t mins_sv_4 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

            q8sums_sv_1 = svld1sh_s32(svptrue_b32(), y[i].bsums + 8);
            q8sums_sv_2 = svld1sh_s32(svptrue_b32(), y[i].bsums + 12);

            svint32_t s1 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), mins_sv_3, q8sums_sv_1), svmul_s32_x(svptrue_b32(), mins_sv_4, q8sums_sv_2));

            svfloat32_t temp = svcvt_f32_s32_x(svptrue_b32(), svadd_s32_x(svptrue_b32(), s0, s1));

            acc_sum = svmla_f32_m(svptrue_b32(), acc_sum, temp, dmin_broad);

            svint32_t sumi1 = svdup_n_s32(0);

            {
                const svuint8_t q2bits_1 = svld1_u8(svptrue_b8(), q2);
                svint8_t q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_1, m3s));
                svint8_t q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;
                const svint32_t scales_sv = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc), m4s));

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 0));

                const svuint8_t q2bits_3 = svld1_u8(svptrue_b8(), q2 + 16);
                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_3, m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 1));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 2), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 2));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 2), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 3));

                const svint32_t scales_sv_1 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc + 4), m4s));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 4), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 0));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 4), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 1));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 6), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 2));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 6), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 3));

                //-------------------------------

                q2 += 32;
                const svint32_t scales_sv_2 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc + 8), m4s));
                const svuint8_t q2bits_2 = svld1_u8(svptrue_b8(), q2);

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_2, m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 0));

                const svuint8_t q2bits_4 = svld1_u8(svptrue_b8(), q2 + 16);
                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_4, m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 1));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 2), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 2));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 2), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 3));

                const svint32_t scales_sv_3 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc + 12), m4s));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 4), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 0));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 4), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 1));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 6), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 2));

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 6), m3s));
                q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 3));
            }
            acc_sum = svmla_f32_m(svptrue_b32(), acc_sum, svcvt_f32_s32_x(svptrue_b32(), sumi1), d_broad);
        }
        *s = svaddv_f32(svptrue_b32(), acc_sum);
        break;

    case 256:
    case 512:
        for (int i = 0; i < nb; ++i) {
            const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
            svfloat32_t d_broad = svdup_n_f32((float32_t)d);
            const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);
            svfloat32_t dmin_broad = svdup_n_f32((float32_t)dmin);

            const uint8_t *__restrict q2 = x[i].qs;
            const int8_t *__restrict q8_sv = y[i].qs;
            const uint8_t *__restrict sc = x[i].scales;

            const svuint32_t mins_and_scales_sve = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc);
            sc += 8;
            const svint32_t scales_sv = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), mins_and_scales_sve, m4s));
            const svint32_t mins_sv_1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), mins_and_scales_sve, 4));
            svint32_t q8sums_sv_1 = svld1sh_s32(svptrue_pat_b32(SV_VL8), y[i].bsums);

            const svuint32_t mins_and_scales_sve_1 = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc);
            const svint32_t scales_sv_1 = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), mins_and_scales_sve_1, m4s));
            const svint32_t mins_sv_2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), mins_and_scales_sve_1, 4));

            svint32_t q8sums_sv_2 = svld1sh_s32(svptrue_pat_b32(SV_VL8), y[i].bsums + 8);

            svfloat32_t temp = svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8), svadd_s32_x(svptrue_pat_b32(SV_VL8), svmul_s32_x(svptrue_pat_b32(SV_VL8), mins_sv_1, q8sums_sv_1), svmul_s32_x(svptrue_pat_b32(SV_VL8), mins_sv_2, q8sums_sv_2)));

            acc_sum = svmla_f32_m(svptrue_pat_b32(SV_VL8), acc_sum, temp, dmin_broad);

            svint32_t sumi1 = svdup_n_s32(0);

            {
                const svuint8_t q2bits_1 = svld1_u8(svptrue_pat_b8(SV_VL32), q2);
                svint8_t q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q2bits_1, m3s));
                svint8_t q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                svint32_t scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv, 0), svdup_lane_s32(scales_sv, 1));
                sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 2), m3s));
                q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                svint32_t scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv, 2), svdup_lane_s32(scales_sv, 3));
                sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(svdup_n_s32(0), q2bytes_sv, q8bytes_sv), scale_2);

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 4), m3s));
                q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv, 4), svdup_lane_s32(scales_sv, 5));
                sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 6), m3s));
                q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv, 6), svdup_lane_s32(scales_sv, 7));
                sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);

                q2 += 32;

                const svuint8_t q2bits_2 = svld1_u8(svptrue_pat_b8(SV_VL32), q2);
                q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q2bits_2, m3s));
                q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 0), svdup_lane_s32(scales_sv_1, 1));
                sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 2), m3s));
                q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 2), svdup_lane_s32(scales_sv_1, 3));
                sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 4), m3s));
                q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 4), svdup_lane_s32(scales_sv_1, 5));
                sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 6), m3s));
                q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 6), svdup_lane_s32(scales_sv_1, 7));
                sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);
            }
            acc_sum = svmla_f32_m(svptrue_pat_b32(SV_VL8), acc_sum, svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8), sumi1), d_broad);
        }
        *s = svaddv_f32(svptrue_pat_b32(SV_VL8), acc_sum);
        break;

    default:
        assert(false && "Unsupported vector length");
        break;
    }

#elif __ARM_NEON
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);

    const int32x4_t vzero = vdupq_n_s32(0);

    mllm_int8x16x2_t q2bytes;
    uint8_t aux[16];

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        const uint8_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        const uint8_t *__restrict sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        vst1q_u8(aux, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const mllm_int16x8x2_t q8sums = mllm_vld1q_s16_x2(y[i].bsums);
        const mllm_int16x8x2_t mins16 = {{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)))}};
        const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16(mins16.val[0]), vget_low_s16(q8sums.val[0])),
                                       vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16(mins16.val[1]), vget_low_s16(q8sums.val[1])),
                                       vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;

        // We use this macro instead of a function call because for some reason
        // the code runs 2-3% slower, even if the function is declared inline
#define MULTIPLY_ACCUM_WITH_SCALE(index)                                                           \
    isum += vaddvq_s32(mllm_vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is + (index)]; \
    isum += vaddvq_s32(mllm_vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is + 1 + (index)];

#define SHIFT_MULTIPLY_ACCUM_WITH_SCALE(shift, index)                                       \
    q8bytes = mllm_vld1q_s8_x2(q8);                                                         \
    q8 += 32;                                                                               \
    q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3)); \
    q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3)); \
    MULTIPLY_ACCUM_WITH_SCALE((index));

        for (int j = 0; j < QK_K / 128; ++j) {
            const mllm_uint8x16x2_t q2bits = mllm_vld1q_u8_x2(q2);
            q2 += 32;

            mllm_int8x16x2_t q8bytes = mllm_vld1q_s8_x2(q8);
            q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));

            MULTIPLY_ACCUM_WITH_SCALE(0);

            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(2, 2);
            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(4, 4);
            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(6, 6);

            is += 8;
        }

        sum += d * isum;
    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        const uint8_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i *)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *)y[i].bsums));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i *)q2);
            q2 += 32;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(0x3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float dall = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        const uint8_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        // load mins and scales from block_q2_K.scales[QK_K/16]
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i *)x[i].scales);
        const __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
        const __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));

        // summs = y[i].bsums * (x[i].scales >> 4) in 16bits*8*2 to 32bits*4*2
        const __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((const __m128i *)&y[i].bsums[0]));
        const __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((const __m128i *)&y[i].bsums[8]));

        // sumf += -dmin * summs in 32bits*8
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(MM256_SET_M128I(summs_1, summs_0))), acc);

        const __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
        const __m128i scales[2] = {scales_0, scales_1};

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K / 128; ++j) {
            // load Q8 quants int8*16*8 from block_q8_K.qs[QK_K]
            const __m128i q8_0 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;

            // load 2bits*16*8 from block_q2_K.qs[QK_K/4]
            __m128i q2bits = _mm_loadu_si128((const __m128i *)q2);
            q2 += 16;
            const __m128i q2_0 = _mm_and_si128(q2bits, m3);
            const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            q2bits = _mm_loadu_si128((const __m128i *)q2);
            q2 += 16;
            const __m128i q2_1 = _mm_and_si128(q2bits, m3);
            const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

            // isuml = q8[l] * ((q2[l] >> shift) & 3) in 8bits*16*8 to 16bits*8*8
            __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
            __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
            __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
            __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
            __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
            __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
            __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
            __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

            // isum += (x[i].scales[is++] & 0xF) * isuml in 16bits*8*8 to 32bits*4*8
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

            p0 = _mm_add_epi32(p0, p1);
            p2 = _mm_add_epi32(p2, p3);
            p4 = _mm_add_epi32(p4, p5);
            p6 = _mm_add_epi32(p6, p7);

            // isum in 32bits*4*2
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
        }

        // sumf += dall * isum - dmin * summs in 32bits
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dall), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __wasm_simd128__
    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        // Vectorized summs calculation
        v128_t summs_vec = wasm_i32x4_splat(0);
        {
            v128_t sc_vec = wasm_v128_load(sc);
            v128_t sc_upper = wasm_u8x16_shr(sc_vec, 4);

            v128_t sc_low = wasm_u16x8_extend_low_u8x16(sc_upper);
            v128_t sc_high = wasm_u16x8_extend_high_u8x16(sc_upper);

            v128_t bsums1 = wasm_v128_load(&y[i].bsums[0]);
            v128_t bsums2 = wasm_v128_load(&y[i].bsums[8]);

            summs_vec = wasm_i32x4_add(
                wasm_i32x4_add(wasm_i32x4_dot_i16x8(sc_low, bsums1),
                               wasm_i32x4_dot_i16x8(sc_high, bsums2)),
                summs_vec);

            summs_vec = wasm_i32x4_add(summs_vec, wasm_i32x4_shuffle(summs_vec, summs_vec, 2, 3, 0, 1));
            summs_vec = wasm_i32x4_add(summs_vec, wasm_i32x4_shuffle(summs_vec, summs_vec, 1, 0, 3, 2));
        }
        int32_t summs = wasm_i32x4_extract_lane(summs_vec, 0);

        // Vectorized isum calculation
        int32_t isum = 0;
        const uint8_t *sc_ptr = sc;
        const int k_iters = QK_K / 128;

        for (int k = 0; k < k_iters; ++k) {
            v128_t isum_vec = wasm_i32x4_splat(0);
            int shift = 0;

            for (int j = 0; j < 4; ++j) {
                const int d0 = (sc_ptr[0] & 0xF);
                const int d1 = (sc_ptr[1] & 0xF);
                sc_ptr += 2;

                // Process first 16 elements
                v128_t q2_0 = wasm_v128_load(q2);
                v128_t q8_0 = wasm_v128_load(q8);
                v128_t q2_shift_0 = wasm_u8x16_shr(q2_0, shift);
                v128_t q2_bits_0 = wasm_v128_and(q2_shift_0, wasm_i8x16_splat(0x03));

                // Process next 16 elements
                v128_t q2_1 = wasm_v128_load(q2 + 16);
                v128_t q8_1 = wasm_v128_load(q8 + 16);
                v128_t q2_shift_1 = wasm_u8x16_shr(q2_1, shift);
                v128_t q2_bits_1 = wasm_v128_and(q2_shift_1, wasm_i8x16_splat(0x03));

                // Calculate dot products
                v128_t p0 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_low_i8x16(q8_0),
                    wasm_i16x8_extend_low_i8x16(q2_bits_0));
                v128_t p1 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_high_i8x16(q8_0),
                    wasm_i16x8_extend_high_i8x16(q2_bits_0));
                v128_t p2 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_low_i8x16(q8_1),
                    wasm_i16x8_extend_low_i8x16(q2_bits_1));
                v128_t p3 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_high_i8x16(q8_1),
                    wasm_i16x8_extend_high_i8x16(q2_bits_1));

                // Accumulate scaled results
                v128_t scaled = wasm_i32x4_add(
                    wasm_i32x4_mul(wasm_i32x4_add(p0, p1), wasm_i32x4_splat(d0)),
                    wasm_i32x4_mul(wasm_i32x4_add(p2, p3), wasm_i32x4_splat(d1)));

                isum_vec = wasm_i32x4_add(isum_vec, scaled);
                q8 += 32;
                shift += 2;
            }
            q2 += 32;

            // Horizontal sum of isum_vec
            isum_vec = wasm_i32x4_add(isum_vec, wasm_i32x4_shuffle(isum_vec, isum_vec, 2, 3, 0, 1));
            isum_vec = wasm_i32x4_add(isum_vec, wasm_i32x4_shuffle(isum_vec, isum_vec, 1, 0, 3, 2));
            isum += wasm_i32x4_extract_lane(isum_vec, 0);
        }

        const float dall = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        const float dmin = MLLM_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf += dall * isum - dmin * summs;
    }

    *s = sumf;

#elif defined __riscv_v_intrinsic

    const int vector_length = __riscv_vlenb() * 8;
    float sumf = 0;

    uint8_t temp_01[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    uint8_t atmp[16];

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {
            const uint8_t *q2 = x[i].qs;
            const int8_t *q8 = y[i].qs;
            const uint8_t *sc = x[i].scales;

            const float dall = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
            const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

            size_t vl = 16;

            vuint8m1_t scales = __riscv_vle8_v_u8m1(sc, vl);
            vuint8m1_t aux = __riscv_vand_vx_u8m1(scales, 0x0F, vl);

            vint16m1_t q8sums = __riscv_vle16_v_i16m1(y[i].bsums, vl);

            vuint8mf2_t scales_2 = __riscv_vle8_v_u8mf2(sc, vl);
            vuint8mf2_t mins8 = __riscv_vsrl_vx_u8mf2(scales_2, 0x4, vl);
            vint16m1_t mins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
            vint32m2_t prod = __riscv_vwmul_vv_i32m2(q8sums, mins, vl);
            vint32m1_t vsums = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);

            sumf += dmin * __riscv_vmv_x_s_i32m1_i32(vsums);

            vl = 32;

            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vuint8m1_t v_b = __riscv_vle8_v_u8m1(temp_01, vl);

            uint8_t is = 0;
            int isum = 0;

            for (int j = 0; j < QK_K / 128; ++j) {
                // load Q2
                vuint8m1_t q2_x = __riscv_vle8_v_u8m1(q2, vl);

                vuint8m1_t q2_0 = __riscv_vand_vx_u8m1(q2_x, 0x03, vl);
                vuint8m1_t q2_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x2, vl), 0x03, vl);
                vuint8m1_t q2_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x4, vl), 0x03, vl);
                vuint8m1_t q2_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x6, vl), 0x03, vl);

                // duplicate scale elements for product
                vuint8m1_t sc0 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 0 + is, vl), vl);
                vuint8m1_t sc1 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 2 + is, vl), vl);
                vuint8m1_t sc2 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 4 + is, vl), vl);
                vuint8m1_t sc3 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 6 + is, vl), vl);

                vint16m2_t p0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_0, sc0, vl));
                vint16m2_t p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_1, sc1, vl));
                vint16m2_t p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_2, sc2, vl));
                vint16m2_t p3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_3, sc3, vl));

                // load Q8
                vint8m1_t q8_0 = __riscv_vle8_v_i8m1(q8, vl);
                vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8 + 32, vl);
                vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8 + 64, vl);
                vint8m1_t q8_3 = __riscv_vle8_v_i8m1(q8 + 96, vl);

                vint32m4_t s0 = __riscv_vwmul_vv_i32m4(p0, __riscv_vwcvt_x_x_v_i16m2(q8_0, vl), vl);
                vint32m4_t s1 = __riscv_vwmul_vv_i32m4(p1, __riscv_vwcvt_x_x_v_i16m2(q8_1, vl), vl);
                vint32m4_t s2 = __riscv_vwmul_vv_i32m4(p2, __riscv_vwcvt_x_x_v_i16m2(q8_2, vl), vl);
                vint32m4_t s3 = __riscv_vwmul_vv_i32m4(p3, __riscv_vwcvt_x_x_v_i16m2(q8_3, vl), vl);

                vint32m1_t isum0 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s0, s1, vl), vzero, vl);
                vint32m1_t isum1 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s2, s3, vl), isum0, vl);

                isum += __riscv_vmv_x_s_i32m1_i32(isum1);

                q2 += 32;
                q8 += 128;
                is = 8;
            }

            sumf += dall * isum;
        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const uint8_t *q2 = x[i].qs;
            const int8_t *q8 = y[i].qs;
            const uint8_t *sc = x[i].scales;
            const float dall = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
            const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);
            uint8_t *patmp = atmp;
            int vsums;
            int tmp;
            __asm__ __volatile__(
                "vsetivli zero, 16, e8, m1\n\t"
                "vmv.v.x v8, zero\n\t"
                "vle8.v v1, (%[sc])\n\t"
                "vand.vi v0, v1, 0xF\n\t"
                "vsrl.vi v1, v1, 4\n\t"
                "vse8.v v0, (%[scale])\n\t"
                "vsetivli zero, 16, e16, m2\n\t"
                "vle16.v v2, (%[bsums])\n\t"
                "vzext.vf2 v0, v1\n\t"
                "vwmul.vv v4, v0, v2\n\t"
                "vsetivli zero, 16, e32, m4\n\t"
                "vredsum.vs v8, v4, v8\n\t"
                "vmv.x.s %[vsums], v8"
                : [tmp] "=&r"(tmp), [vsums] "=&r"(vsums)
                : [sc] "r"(sc), [scale] "r"(atmp), [bsums] "r"(y[i].bsums)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            sumf += dmin * vsums;
            int isum = 0;

            for (int j = 0; j < QK_K / 128; ++j) {
                __asm__ __volatile__(
                    "vsetvli zero, %[vl32], e8, m2\n\t"
                    "vle8.v v0, (%[q2])\n\t"
                    "vsrl.vi v2, v0, 2\n\t"
                    "vsrl.vi v4, v0, 4\n\t"
                    "vsrl.vi v6, v0, 6\n\t"
                    "vand.vi v0, v0, 0x3\n\t"
                    "vand.vi v2, v2, 0x3\n\t"
                    "vand.vi v4, v4, 0x3\n\t"
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vle8.v v8, (%[q8])\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vwmul.vv v24, v4, v12\n\t"
                    "vsetivli zero, 16, e16, m2\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "vwredsum.vs v10, v16, v0\n\t"
                    "vwredsum.vs v9, v18, v0\n\t"
                    "vwredsum.vs v8, v20, v0\n\t"
                    "vwredsum.vs v7, v22, v0\n\t"
                    "vwredsum.vs v11, v24, v0\n\t"
                    "vwredsum.vs v12, v26, v0\n\t"
                    "vwredsum.vs v13, v28, v0\n\t"
                    "vwredsum.vs v14, v30, v0\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vslideup.vi v10, v9, 1\n\t"
                    "vslideup.vi v8, v7, 1\n\t"
                    "vslideup.vi v11, v12, 1\n\t"
                    "vslideup.vi v13, v14, 1\n\t"
                    "vslideup.vi v10, v8, 2\n\t"
                    "vslideup.vi v11, v13, 2\n\t"
                    "vsetivli zero, 8, e32, m2\n\t"
                    "vle8.v v15, (%[scale])\n\t"
                    "vzext.vf4 v12, v15\n\t"
                    "vmul.vv v10, v10, v12\n\t"
                    "vredsum.vs v0, v10, v0\n\t"
                    "vmv.x.s %[tmp], v0\n\t"
                    "add %[isum], %[isum], %[tmp]"
                    : [tmp] "=&r"(tmp), [isum] "+&r"(isum)
                    : [q2] "r"(q2), [scale] "r"(patmp), [q8] "r"(q8), [vl32] "r"(32), [vl64] "r"(64), [vl128] "r"(128)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                q2 += 32;
                q8 += 128;
                patmp += 8;
            }

            sumf += dall * isum;
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0x3);
    const vector signed char lowScaleMask = vec_splats((signed char)0xF);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v2 = vec_splats((unsigned char)0x2);
    const vector unsigned char v6 = vec_splats((unsigned char)0x6);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(MLLM_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector float vxmin = vec_splats(MLLM_FP16_TO_FP32(x[i].dmin));
        vector float vdmin = vec_mul(vxmin, vyd);

        vector signed short q8ysums0 = vec_xl(0, y[i].bsums);
        vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

        vector signed char q2xmins = (vector signed char)vec_xl(0, x[i].scales);
        vector signed char vscales = vec_and(q2xmins, lowScaleMask);

        q2xmins = vec_sr(q2xmins, v4);
        vector signed short q2xmins0 = vec_unpackh(q2xmins);
        vector signed short q2xmins1 = vec_unpackl(q2xmins);

        vector signed int prod0 = vec_mule(q2xmins0, q8ysums0);
        vector signed int prod1 = vec_mulo(q2xmins0, q8ysums0);
        vector signed int prod2 = vec_mule(q2xmins1, q8ysums1);
        vector signed int prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;
        vector signed int vsumi4 = v0;
        vector signed int vsumi5 = v0;
        vector signed int vsumi6 = v0;
        vector signed int vsumi7 = v0;

        const uint8_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K / 128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl(0, q2);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q2);
            q2 += 32;

            vector unsigned char q2x00 = (vector unsigned char)vec_and(qxs0, lowMask);
            vector unsigned char q2x01 = (vector unsigned char)vec_and(vec_sr(qxs0, v2), lowMask);
            vector unsigned char q2x02 = (vector unsigned char)vec_and(vec_sr(qxs0, v4), lowMask);
            vector unsigned char q2x03 = (vector unsigned char)vec_and(vec_sr(qxs0, v6), lowMask);
            vector unsigned char q2x10 = (vector unsigned char)vec_and(qxs1, lowMask);
            vector unsigned char q2x11 = (vector unsigned char)vec_and(vec_sr(qxs1, v2), lowMask);
            vector unsigned char q2x12 = (vector unsigned char)vec_and(vec_sr(qxs1, v4), lowMask);
            vector unsigned char q2x13 = (vector unsigned char)vec_and(vec_sr(qxs1, v6), lowMask);

            vector signed char q8y00 = vec_xl(0, q8);
            vector signed char q8y10 = vec_xl(16, q8);
            vector signed char q8y01 = vec_xl(32, q8);
            vector signed char q8y11 = vec_xl(48, q8);
            vector signed char q8y02 = vec_xl(64, q8);
            vector signed char q8y12 = vec_xl(80, q8);
            vector signed char q8y03 = vec_xl(96, q8);
            vector signed char q8y13 = vec_xl(112, q8);
            q8 += 128;

            vector signed int qv0 = vec_msum(q8y00, q2x00, v0);
            vector signed int qv1 = vec_msum(q8y01, q2x01, v0);
            vector signed int qv2 = vec_msum(q8y02, q2x02, v0);
            vector signed int qv3 = vec_msum(q8y03, q2x03, v0);
            vector signed int qv4 = vec_msum(q8y10, q2x10, v0);
            vector signed int qv5 = vec_msum(q8y11, q2x11, v0);
            vector signed int qv6 = vec_msum(q8y12, q2x12, v0);
            vector signed int qv7 = vec_msum(q8y13, q2x13, v0);

            vector signed short vscales_07 = vec_unpackh(vscales);
            vector signed int vscales_03 = vec_unpackh(vscales_07);
            vector signed int vscales_47 = vec_unpackl(vscales_07);
            vector signed int vs0 = vec_splat(vscales_03, 0);
            vector signed int vs1 = vec_splat(vscales_03, 1);
            vector signed int vs2 = vec_splat(vscales_03, 2);
            vector signed int vs3 = vec_splat(vscales_03, 3);
            vector signed int vs4 = vec_splat(vscales_47, 0);
            vector signed int vs5 = vec_splat(vscales_47, 1);
            vector signed int vs6 = vec_splat(vscales_47, 2);
            vector signed int vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_add(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_add(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_add(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_add(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_add(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_add(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_add(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_add(vsumi0, vsumi4);
        vsumi1 = vec_add(vsumi1, vsumi5);
        vsumi2 = vec_add(vsumi2, vsumi6);
        vsumi3 = vec_add(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx

    __m256 acc = (__m256)__lasx_xvldi(0);

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        const uint8_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        const __m128i mins_and_scales128 = __lsx_vld((const __m128i *)x[i].scales, 0);
        const __m128i scales128 = __lsx_vandi_b(mins_and_scales128, 0xf);
        const __m256i mins = lasx_ext8_16(__lsx_vsrli_b(mins_and_scales128, 4));
        const __m256i prod = lasx_madd_h(mins, __lasx_xvld((const __m256i *)y[i].bsums, 0));

        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(dmin), __lasx_xvffint_s_w(prod), acc);

        const v16i8 shuffle_mask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        const __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_mask));

        __m256i sumi = __lasx_xvldi(0);

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q2bits = __lasx_xvld((const __m256i *)q2, 0);
            q2 += 32;

            const __m256i q8_0 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;
            const __m256i q8_3 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;

            const __m256i q2_0 = __lasx_xvandi_b(q2bits, 3);
            const __m256i q2_1 = __lasx_xvandi_b(__lasx_xvsrli_b(q2bits, 2), 3);
            const __m256i q2_2 = __lasx_xvandi_b(__lasx_xvsrli_b(q2bits, 4), 3);
            const __m256i q2_3 = __lasx_xvsrli_b(q2bits, 6);

            __m256i p0 = lasx_madd_h_b(q2_0, q8_0);
            __m256i p1 = lasx_madd_h_b(q2_1, q8_1);
            __m256i p2 = lasx_madd_h_b(q2_2, q8_2);
            __m256i p3 = lasx_madd_h_b(q2_3, q8_3);

            p0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p0);
            p1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p1);
            p2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p2);
            p3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p3);

            p0 = __lasx_xvadd_w(p0, p1);
            p2 = __lasx_xvadd_w(p2, p3);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p0, p2));
        }

        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);
    }

    *s = hsum_float_8(acc);

#else

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K / 128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
#endif
}

void vec_dot_iq2_xxs_q8_K(int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    assert(n % QK_K == 0);

    const block_iq2_xxs *__restrict x = (block_iq2_xxs *)vx;
    const block_q8_K *__restrict y = (block_q8_K *)vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t *signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t *aux8 = (const uint8_t *)aux32;

    mllm_int8x16x4_t q2u;
    mllm_int8x16x4_t q2s;
    mllm_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        float sumf1 = 0, sumf2 = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            q8b = mllm_vld1q_s8_x4(q8);
            q8 += 64;
            memcpy(aux32, q2, 4 * sizeof(uint32_t));
            q2 += 8;
            q2u.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[0])), vld1_s8((const int8_t *)(iq2xxs_grid + aux8[1])));
            q2u.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[2])), vld1_s8((const int8_t *)(iq2xxs_grid + aux8[3])));
            q2u.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[8])), vld1_s8((const int8_t *)(iq2xxs_grid + aux8[9])));
            q2u.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[10])), vld1_s8((const int8_t *)(iq2xxs_grid + aux8[11])));
            q2s.val[0] = vcombine_s8(vld1_s8((const int8_t *)(signs64 + ((aux32[1] >> 0) & 127))), vld1_s8((const int8_t *)(signs64 + ((aux32[1] >> 7) & 127))));
            q2s.val[1] = vcombine_s8(vld1_s8((const int8_t *)(signs64 + ((aux32[1] >> 14) & 127))), vld1_s8((const int8_t *)(signs64 + ((aux32[1] >> 21) & 127))));
            q2s.val[2] = vcombine_s8(vld1_s8((const int8_t *)(signs64 + ((aux32[3] >> 0) & 127))), vld1_s8((const int8_t *)(signs64 + ((aux32[3] >> 7) & 127))));
            q2s.val[3] = vcombine_s8(vld1_s8((const int8_t *)(signs64 + ((aux32[3] >> 14) & 127))), vld1_s8((const int8_t *)(signs64 + ((aux32[3] >> 21) & 127))));
            q2u.val[0] = vmulq_s8(q2u.val[0], q2s.val[0]);
            q2u.val[1] = vmulq_s8(q2u.val[1], q2s.val[1]);
            q2u.val[2] = vmulq_s8(q2u.val[2], q2s.val[2]);
            q2u.val[3] = vmulq_s8(q2u.val[3], q2s.val[3]);
            const int32x4_t p1 = mllm_vdotq_s32(mllm_vdotq_s32(vdupq_n_s32(0), q2u.val[0], q8b.val[0]), q2u.val[1], q8b.val[1]);
            const int32x4_t p2 = mllm_vdotq_s32(mllm_vdotq_s32(vdupq_n_s32(0), q2u.val[2], q8b.val[2]), q2u.val[3], q8b.val[3]);
            sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[1] >> 28));
            sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[3] >> 28));
        }
        sumf += d * (sumf1 + sumf2);
    }
    *s = 0.25f * sumf;

#elif defined(__AVX2__)

    const uint64_t *signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t *aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            memcpy(aux32, q2, 4 * sizeof(uint32_t));
            q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const uint64_t *signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t *aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            memcpy(aux32, q2, 4 * sizeof(uint32_t));
            q2 += 8;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]]);
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127]);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2 * ls1 + 1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2 * ls1 + 1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2 * ls2 + 1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2 * ls2 + 1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
    const vector int v0 = vec_splats((int32_t)0);
    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const uint64_t *signs64 = (const uint64_t *)keven_signs_q2xs;

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(MLLM_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint16_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K / 32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            const uint8_t *aux8 = (const uint8_t *)aux32;

            memcpy(aux32, q2, 4 * sizeof(uint32_t));
            q2 += 8;

            vector signed long long aux64x2_0 = {*(const int64_t *)(iq2xxs_grid + aux8[0]), *(const int64_t *)(iq2xxs_grid + aux8[1])};
            vector signed long long aux64x2_1 = {*(const int64_t *)(iq2xxs_grid + aux8[2]), *(const int64_t *)(iq2xxs_grid + aux8[3])};
            vector signed long long aux64x2_2 = {*(const int64_t *)(iq2xxs_grid + aux8[8]), *(const int64_t *)(iq2xxs_grid + aux8[9])};
            vector signed long long aux64x2_3 = {*(const int64_t *)(iq2xxs_grid + aux8[10]), *(const int64_t *)(iq2xxs_grid + aux8[11])};

            vector signed long long vsigns0 = {*(const int64_t *)(signs64 + ((aux32[1] >> 0) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >> 7) & 127))};
            vector signed long long vsigns1 = {*(const int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            vector signed long long vsigns2 = {*(const int64_t *)(signs64 + ((aux32[3] >> 0) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >> 7) & 127))};
            vector signed long long vsigns3 = {*(const int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            vector signed char q2x0 = (vector signed char)vec_mul((vector signed char)vsigns0, (vector signed char)aux64x2_0);
            vector signed char q2x1 = (vector signed char)vec_mul((vector signed char)vsigns1, (vector signed char)aux64x2_1);
            vector signed char q2x2 = (vector signed char)vec_mul((vector signed char)vsigns2, (vector signed char)aux64x2_2);
            vector signed char q2x3 = (vector signed char)vec_mul((vector signed char)vsigns3, (vector signed char)aux64x2_3);

            vector signed char q8y0 = vec_xl(0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            const uint16_t ls0 = aux32[1] >> 28;
            const uint16_t ls1 = aux32[3] >> 28;

            vector signed short vscales01 = vec_splats((int16_t)(2 * ls0 + 1));
            vector signed short vscales23 = vec_splats((int16_t)(2 * ls1 + 1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

    const uint64_t *signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t *aux8 = (const uint8_t *)aux32;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;
            memcpy(aux32, q2, 4 * sizeof(uint32_t));
            q2 += 8;

            const __m256i q2_1 = lasx_set_d(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = lasx_set_d(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = lasx_set_d(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                            signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m256i s2_2 = lasx_set_d(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                            signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const __m256i q8s_1 = __lasx_xvsigncov_b(s2_1, q8_1);
            const __m256i q8s_2 = __lasx_xvsigncov_b(s2_2, q8_2);
            const __m256i dot1 = lasx_maddubs_h(q2_1, q8s_1);
            const __m256i dot2 = lasx_maddubs_h(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2 * ls1 + 1));
            const __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2 * ls2 + 1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);
// #elif defined(__VXE__) || defined(__VXE2__)
//     const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;
//
//     uint32_t aux32[4];
//     const uint8_t * aux8 = (const uint8_t *)aux32;
//
//     float sumf = 0;
//
//     for (int i = 0; i < nb; ++i) {
//         const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
//         const uint16_t * __restrict q2 = x[i].qs;
//         const int8_t   * __restrict q8 = y[i].qs;
//
//         float sumf1 = 0, sumf2 = 0;
//
//         for (int ib32 = 0; ib32 < QK_K/32; ib += 2) {
//             int8x16_t q8b0 = vec_xl( 0, q8);
//             int8x16_t qb81 = vec_xl(16, q8);
//             int8x16_t q8b2 = vec_xl(32, q8);
//             int8x16_t q8b3 = vec_xl(48, q8);
//             q8 += 64;
//
//             memcpy(aux32, q2, 4 * sizeof(uint32_t));
//             q2 += 8;
//
//             int8x16_t q2u0 = { *(const int64_t *)(iq2xxs_grid + aux8[ 0]), *(const int64_t *)(iq2xxs_grid + aux8[ 1]) };
//             int8x16_t q2u1 = { *(const int64_t *)(iq2xxs_grid + aux8[ 2]), *(const int64_t *)(iq2xxs_grid + aux8[ 3]) };
//             int8x16_t q2u2 = { *(const int64_t *)(iq2xxs_grid + aux8[ 8]), *(const int64_t *)(iq2xxs_grid + aux8[ 9]) };
//             int8x16_t q2u3 = { *(const int64_t *)(iq2xxs_grid + aux8[10]), *(const int64_t *)(iq2xxs_grid + aux8[11]) };
//
//             int8x16_t q2s0 = { *(const int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >>  7) & 127)) };
//             int8x16_t q2s1 = { *(const int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >> 21) & 127)) };
//             int8x16_t q2s2 = { *(const int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >>  7) & 127)) };
//             int8x16_t q2s3 = { *(const int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >> 21) & 127)) };
//
//             q2u0 = vec_mul(q2u0, q2s0);
//             q2u1 = vec_mul(q2u1, q2s1);
//             q2u2 = vec_mul(q2u2, q2s2);
//             q2u3 = vec_mul(q2u3, q2s3);
//
//             const int32x4_t p1 = mllm_vec_dot(mllm_vec_dot(vec_splat_s32(0), q2u0, q8b0), q2u1, q8b1);
//             const int32x4_t p2 = mllm_vec_dot(mllm_vec_dot(vec_splat_s32(0), q2u2, q8b2), q2u3, q8b3);
//
//             sumf1 += (p1[0] + p1[1] + p1[2] + p1[3]) * (0.5f + (aux32[1] >> 28));
//             sumf2 += (p2[0] + p2[1] + p2[2] + p2[3]) * (0.5f + (aux32[3] >> 28));
//         }
//
//         sumf += d * (sumf1 + sumf2);
//     }
//
//     *s = 0.25f * sumf;
#else

    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t *__restrict q2 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
#endif
}