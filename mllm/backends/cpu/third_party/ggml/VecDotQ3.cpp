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

#include "VecDotQ3.hpp"
#include "ComputeUtils.hpp"

void vec_dot_q3_K_q8_K(int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    assert(n % QK_K == 0);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K *__restrict x = (block_q3_K *)vx;
    const block_q8_K *__restrict y = (block_q8_K *)vy;

    const int nb = n / QK_K;

#if defined(__ARM_FEATURE_SVE)

    uint32_t aux[3];
    uint32_t utmp[4];

    const int8_t m32 = 32;
    const int vector_length = svcntb() * 8;
    const svuint8_t m3b_sv = svdup_n_u8(0x3);
    const svint32_t vzero_sv = svdup_n_s32(0);

    const svuint8_t m0_sv = svdup_n_u8(1);
    const svuint8_t m1_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 1);
    const svuint8_t m2_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 2);
    const svuint8_t m3_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 3);

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q3_sv = x[i].qs;
        const uint8_t *__restrict qh_sv = x[i].hmask;
        const int8_t *__restrict q8_sv = y[i].qs;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t *scale = (int8_t *)utmp;

        for (int j = 0; j < 16; ++j) scale[j] -= m32;

        switch (vector_length) {
        case 128: {
            svuint8_t qhbits_sv_1 = svld1_u8(svptrue_b8(), qh_sv);
            svuint8_t qhbits_sv_2 = svld1_u8(svptrue_b8(), qh_sv + 16);
            svuint8_t q3h_sv;

            svint32_t sumi1_1 = svdup_n_s32(0);
            svint8_t q3bytes_sv;

            for (int j = 0; j < QK_K / 128; ++j) {
                const svuint8_t q3bits_sv = svld1_u8(svptrue_b8(), q3_sv);
                q3_sv += 16;
                const svuint8_t q3bits_sv_1 = svld1_u8(svptrue_b8(), q3_sv);
                q3_sv += 16;
                svint8_t q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;
                svint8_t q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m0_sv, qhbits_sv_1), 2);
                q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), q3bits_sv, m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), svdup_n_s32((int32_t)scale[0]));

                q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m0_sv, qhbits_sv_2), 2);
                q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), q3bits_sv_1, m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), svdup_n_s32((int32_t)scale[1]));

                q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;
                q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m1_sv, qhbits_sv_1), 1);
                q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 2), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), svdup_n_s32((int32_t)scale[2]));

                q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m1_sv, qhbits_sv_2), 1);
                q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 2), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), svdup_n_s32((int32_t)scale[3]));

                scale += 4;
                q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;
                q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                q3h_sv = svbic_u8_x(svptrue_b8(), m2_sv, qhbits_sv_1);
                q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 4), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), svdup_n_s32((int32_t)scale[0]));

                q3h_sv = svbic_u8_x(svptrue_b8(), m2_sv, qhbits_sv_2);
                q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 4), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), svdup_n_s32((int32_t)scale[1]));

                q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;
                q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv);
                q8_sv += 16;

                q3h_sv = svlsr_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m3_sv, qhbits_sv_1), 1);
                q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 6), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), svdup_n_s32((int32_t)scale[2]));

                q3h_sv = svlsr_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m3_sv, qhbits_sv_2), 1);
                q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 6), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), svdup_n_s32((int32_t)scale[3]));

                if (j == 0) {
                    qhbits_sv_1 = svlsr_n_u8_x(svptrue_b8(), qhbits_sv_1, 4);
                    qhbits_sv_2 = svlsr_n_u8_x(svptrue_b8(), qhbits_sv_2, 4);
                }

                scale += 4;
            }

            sum += d * (svaddv_s32(svptrue_b32(), sumi1_1));
        } break;
        case 256:
        case 512: {
            svuint8_t qhbits_sv = svld1_u8(svptrue_pat_b8(SV_VL32), qh_sv);
            svuint8_t q3h_sv;

            svint32_t sumi1_1 = svdup_n_s32(0);
            svint8_t q3bytes_sv;

            for (int j = 0; j < QK_K / 128; ++j) {
                const svuint8_t q3bits_sv = svld1_u8(svptrue_pat_b8(SV_VL32), q3_sv);
                q3_sv += 32;
                svint8_t q8bytes_1_sv_1 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;
                svint8_t q8bytes_1_sv_2 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                q3h_sv = svlsl_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m0_sv, qhbits_sv), 2);
                q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q3bits_sv, m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                svint32_t scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[0]), svdup_n_s32((int32_t)scale[1]));
                sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), scale_1);

                q3h_sv = svlsl_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m1_sv, qhbits_sv), 1);
                q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 2), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[2]), svdup_n_s32((int32_t)scale[3]));
                sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), scale_1);

                scale += 4;
                q8bytes_1_sv_1 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;
                q8bytes_1_sv_2 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
                q8_sv += 32;

                q3h_sv = svbic_u8_x(svptrue_pat_b8(SV_VL32), m2_sv, qhbits_sv);
                q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 4), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[0]), svdup_n_s32((int32_t)scale[1]));
                sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), scale_1);

                q3h_sv = svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m3_sv, qhbits_sv), 1);
                q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 6), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[2]), svdup_n_s32((int32_t)scale[3]));
                sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), scale_1);

                if (j == 0) {
                    qhbits_sv = svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), qhbits_sv, 4);
                }

                scale += 4;
            }

            sum += d * (svaddv_s32(svptrue_pat_b32(SV_VL8), sumi1_1));
        } break;
        default:
            assert(false && "Unsupported vector length");
            break;
        }
    }
    *s = sum;

#elif __ARM_NEON

    uint32_t aux[3];
    uint32_t utmp[4];

    const uint8x16_t m3b = vdupq_n_u8(0x3);
    const int32x4_t vzero = vdupq_n_s32(0);

    const uint8x16_t m0 = vdupq_n_u8(1);
    const uint8x16_t m1 = vshlq_n_u8(m0, 1);
    const uint8x16_t m2 = vshlq_n_u8(m0, 2);
    const uint8x16_t m3 = vshlq_n_u8(m0, 3);
    const int8_t m32 = 32;

    mllm_int8x16x4_t q3bytes;

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q3 = x[i].qs;
        const uint8_t *__restrict qh = x[i].hmask;
        const int8_t *__restrict q8 = y[i].qs;

        mllm_uint8x16x2_t qhbits = mllm_vld1q_u8_x2(qh);

        mllm_uint8x16x4_t q3h;

        int32_t isum = 0;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t *scale = (int8_t *)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= m32;

        for (int j = 0; j < QK_K / 128; ++j) {
            const mllm_uint8x16x2_t q3bits = mllm_vld1q_u8_x2(q3);
            q3 += 32;
            const mllm_int8x16x4_t q8bytes_1 = mllm_vld1q_s8_x4(q8);
            q8 += 64;
            const mllm_int8x16x4_t q8bytes_2 = mllm_vld1q_s8_x4(q8);
            q8 += 64;

            q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
            q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
            q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
            q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

            isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[0], q8bytes_1.val[0])) * scale[0];
            isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[1], q8bytes_1.val[1])) * scale[1];
            isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[2], q8bytes_1.val[2])) * scale[2];
            isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[3], q8bytes_1.val[3])) * scale[3];

            scale += 4;

            q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
            q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
            q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
            q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)), vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)), vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

            isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[0], q8bytes_2.val[0])) * scale[0];
            isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[1], q8bytes_2.val[1])) * scale[1];
            isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[2], q8bytes_2.val[2])) * scale[2];
            isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[3], q8bytes_2.val[3])) * scale[3];

            scale += 4;

            if (j == 0) {
                qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
                qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
            }
        }
        sum += d * isum;
    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q3 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
            ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
            ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
            (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
            (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        // high bit
        const __m256i hbits = _mm256_loadu_si256((const __m256i *)x[i].hmask);

        // integer accumulator
        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            // load low 2 bits
            const __m256i q3bits = _mm256_loadu_si256((const __m256i *)q3);
            q3 += 32;

            // prepare low and high bits
            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            // load Q8 quants
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            // multiply with scales
            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

            // accumulate
            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
        }

        // multiply with block scale and accumulate
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i mone = _mm_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    const uint32_t *aux;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q3 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        // Set up scales
        aux = (const uint32_t *)x[i].scales;
        __m128i scales128 = _mm_set_epi32(
            ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
            ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
            (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
            (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m128i scales_0 = _mm_cvtepi8_epi16(scales128);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales128, scales128));
        const __m128i scales[2] = {scales_0, scales_1};

        // high bit *128*2 from block_q3_K.hmask[QK_K/8]
        const __m128i hbits_0 = _mm_loadu_si128((const __m128i *)&x[i].hmask[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i *)&x[i].hmask[16]);

        // integer accumulator
        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K / 128; ++j) {
            // load low 2 bits *64*2 from block_q3_K.qs[QK_K/4]
            const __m128i q3bits_0 = _mm_loadu_si128((const __m128i *)q3);
            q3 += 16;
            const __m128i q3bits_1 = _mm_loadu_si128((const __m128i *)q3);
            q3 += 16;

            // prepare low and high bits
            const int bit = j << 2;

            const __m128i q3l_0 = _mm_and_si128(q3bits_0, m3);
            const __m128i q3l_1 = _mm_and_si128(q3bits_1, m3);
            const __m128i q3h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit)), bit), 2);
            const __m128i q3h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit)), bit), 2);

            const __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), m3);
            const __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), m3);
            const __m128i q3h_2 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit + 1)), bit + 1), 2);
            const __m128i q3h_3 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit + 1)), bit + 1), 2);

            const __m128i q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), m3);
            const __m128i q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), m3);
            const __m128i q3h_4 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit + 2)), bit + 2), 2);
            const __m128i q3h_5 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit + 2)), bit + 2), 2);

            const __m128i q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), m3);
            const __m128i q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), m3);
            const __m128i q3h_6 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit + 3)), bit + 3), 2);
            const __m128i q3h_7 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit + 3)), bit + 3), 2);

            // load Q8 quants from block_q8_K.qs[QK_K]
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

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, q8_0);
            __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, q8_1);
            __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, q8_2);
            __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, q8_3);
            __m128i q8s_4 = _mm_maddubs_epi16(q3h_4, q8_4);
            __m128i q8s_5 = _mm_maddubs_epi16(q3h_5, q8_5);
            __m128i q8s_6 = _mm_maddubs_epi16(q3h_6, q8_6);
            __m128i q8s_7 = _mm_maddubs_epi16(q3h_7, q8_7);

            __m128i p16_0 = _mm_maddubs_epi16(q3l_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q3l_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q3l_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q3l_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q3l_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q3l_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q3l_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q3l_7, q8_7);

            p16_0 = _mm_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm_sub_epi16(p16_3, q8s_3);
            p16_4 = _mm_sub_epi16(p16_4, q8s_4);
            p16_5 = _mm_sub_epi16(p16_5, q8s_5);
            p16_6 = _mm_sub_epi16(p16_6, q8s_6);
            p16_7 = _mm_sub_epi16(p16_7, q8s_7);

            // multiply with scales
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p16_0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_7);

            // accumulate
            p16_0 = _mm_add_epi32(p16_0, p16_1);
            p16_2 = _mm_add_epi32(p16_2, p16_3);
            p16_4 = _mm_add_epi32(p16_4, p16_5);
            p16_6 = _mm_add_epi32(p16_6, p16_7);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_4, p16_6));
        }

        // multiply with block scale and accumulate
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __wasm_simd128__
    int8_t aux8[QK_K];
    float sums[8] = {0};
    uint32_t auxs[4];

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *__restrict q3 = x[i].qs;
        const uint8_t *__restrict hm = x[i].hmask;
        const int8_t *__restrict q8 = y[i].qs;

        // Process blocks with SIMD
        int8_t *a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int shift = 0; shift <= 6; shift += 2) {
                v128_t v_m = wasm_i8x16_splat(m);
                for (int l = 0; l < 32; l += 16) {
                    v128_t v_q3 = wasm_v128_load(q3 + l);
                    v128_t v_shift = wasm_i8x16_shr(v_q3, shift);
                    v128_t v_low2 = wasm_v128_and(v_shift, wasm_i8x16_splat(0x03));

                    v128_t v_hm = wasm_v128_load(hm + l);
                    v128_t v_mask = wasm_v128_and(v_hm, v_m);
                    v_mask = wasm_i8x16_ne(v_mask, wasm_i8x16_splat(0));

                    v_low2 = wasm_i8x16_sub(v_low2, wasm_v128_and(wasm_i8x16_splat(4), wasm_v128_not(v_mask)));
                    wasm_v128_store(a + l, v_low2);
                }
                a += 32;
                m <<= 1;
            }
            q3 += 32;
        }

        // Extract scales
        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)auxs;

        // SIMD dot product with register accumulators
        v128_t v_acc0 = wasm_i32x4_splat(0);
        v128_t v_acc1 = wasm_i32x4_splat(0);
        a = aux8;
        for (int j = 0; j < QK_K / 16; ++j) {
            const v128_t v_scale = wasm_i16x8_splat(scales[j] - 32);

            // Process 16 elements per iteration
            for (int k = 0; k < 2; ++k) {
                const v128_t v_q8 = wasm_i16x8_load8x8(q8);
                const v128_t v_a = wasm_i16x8_load8x8(a);

                v128_t v_prod = wasm_i16x8_mul(v_q8, v_a);
                v_prod = wasm_i16x8_mul(v_prod, v_scale);

                v_acc0 = wasm_i32x4_add(v_acc0, wasm_i32x4_extend_low_i16x8(v_prod));
                v_acc1 = wasm_i32x4_add(v_acc1, wasm_i32x4_extend_high_i16x8(v_prod));

                q8 += 8;
                a += 8;
            }
        }

        // Accumulate results
        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        const v128_t v_d = wasm_f32x4_splat(d);
        v128_t v_sum = wasm_f32x4_add(
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(v_acc0), v_d),
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(v_acc1), v_d));

        // Accumulate into sums vector
        wasm_v128_store(sums, wasm_f32x4_add(wasm_v128_load(sums), v_sum));
    }

    // Horizontal sum
    v128_t v_sum = wasm_f32x4_add(wasm_v128_load(sums), wasm_v128_load(sums + 4));
    sumf = wasm_f32x4_extract_lane(v_sum, 0) + wasm_f32x4_extract_lane(v_sum, 1) + wasm_f32x4_extract_lane(v_sum, 2) + wasm_f32x4_extract_lane(v_sum, 3);

    *s = sumf;

#elif defined __riscv_v_intrinsic

    uint32_t aux[3];
    uint32_t utmp[4];

    const int vector_length = __riscv_vlenb() * 8;
    float sumf = 0;

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {
            const uint8_t *__restrict q3 = x[i].qs;
            const uint8_t *__restrict qh = x[i].hmask;
            const int8_t *__restrict q8 = y[i].qs;

            memcpy(aux, x[i].scales, 12);
            utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
            utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
            utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
            utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

            int8_t *scale = (int8_t *)utmp;
            for (int j = 0; j < 16; ++j) scale[j] -= 32;

            size_t vl = 32;
            uint8_t m = 1;

            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);

            int sum_t = 0;

            for (int j = 0; j < QK_K; j += 128) {
                vl = 32;

                // load Q3
                vuint8m1_t q3_x = __riscv_vle8_v_u8m1(q3, vl);

                vint8m1_t q3_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q3_x, 0x03, vl));
                vint8m1_t q3_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x2, vl), 0x03, vl));
                vint8m1_t q3_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x4, vl), 0x03, vl));
                vint8m1_t q3_3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x6, vl), 0x03, vl));

                // compute mask for subtraction
                vuint8m1_t qh_m0 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_0 = __riscv_vmseq_vx_u8m1_b8(qh_m0, 0, vl);
                vint8m1_t q3_m0 = __riscv_vsub_vx_i8m1_mu(vmask_0, q3_0, q3_0, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m1 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_1 = __riscv_vmseq_vx_u8m1_b8(qh_m1, 0, vl);
                vint8m1_t q3_m1 = __riscv_vsub_vx_i8m1_mu(vmask_1, q3_1, q3_1, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m2 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_2 = __riscv_vmseq_vx_u8m1_b8(qh_m2, 0, vl);
                vint8m1_t q3_m2 = __riscv_vsub_vx_i8m1_mu(vmask_2, q3_2, q3_2, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m3 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_3 = __riscv_vmseq_vx_u8m1_b8(qh_m3, 0, vl);
                vint8m1_t q3_m3 = __riscv_vsub_vx_i8m1_mu(vmask_3, q3_3, q3_3, 0x4, vl);
                m <<= 1;

                // load Q8 and take product with Q3
                vint16m2_t a0 = __riscv_vwmul_vv_i16m2(q3_m0, __riscv_vle8_v_i8m1(q8, vl), vl);
                vint16m2_t a1 = __riscv_vwmul_vv_i16m2(q3_m1, __riscv_vle8_v_i8m1(q8 + 32, vl), vl);
                vint16m2_t a2 = __riscv_vwmul_vv_i16m2(q3_m2, __riscv_vle8_v_i8m1(q8 + 64, vl), vl);
                vint16m2_t a3 = __riscv_vwmul_vv_i16m2(q3_m3, __riscv_vle8_v_i8m1(q8 + 96, vl), vl);

                vl = 16;

                // retrieve lane to multiply with scale
                vint32m2_t aux0_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 0), (scale[0]), vl);
                vint32m2_t aux0_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 1), (scale[1]), vl);
                vint32m2_t aux1_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 0), (scale[2]), vl);
                vint32m2_t aux1_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 1), (scale[3]), vl);
                vint32m2_t aux2_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 0), (scale[4]), vl);
                vint32m2_t aux2_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 1), (scale[5]), vl);
                vint32m2_t aux3_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 0), (scale[6]), vl);
                vint32m2_t aux3_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 1), (scale[7]), vl);

                vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux0_0, aux0_1, vl), vzero, vl);
                vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux1_0, aux1_1, vl), isum0, vl);
                vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux2_0, aux2_1, vl), isum1, vl);
                vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux3_0, aux3_1, vl), isum2, vl);

                sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

                q3 += 32;
                q8 += 128;
                scale += 8;
            }

            const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;

            sumf += d * sum_t;
        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const uint8_t *restrict q3 = x[i].qs;
            const uint8_t *restrict qh = x[i].hmask;
            const int8_t *restrict q8 = y[i].qs;

            int8_t *scale = (int8_t *)utmp;
            int tmp;
            __asm__ __volatile__(
                "vsetivli zero, 12, e8, m1\n\t"
                "vle8.v v0, (%[s6b])\n\t"
                "vmv1r.v v2, v0\n\t"
                "vsetivli zero, 2, e64, m1\n\t"
                "vmv.v.x v9, %[sh]\n\t"
                "vslidedown.vi v1, v0, 1\n\t"
                "vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
                "vslideup.vi v0, v2, 1\n\t"     // {aux[0], aux[1], aux[0], aux[1]}
                "vsetivli zero, 4, e32, m1\n\t"
                "vid.v v9\n\t"
                "vmv.x.s %[tmp], v1\n\t"
                "vsll.vi v9, v9, 1\n\t"  // {0, 2, 4, 6}
                "vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
                "vsrl.vv v4, v1, v9\n\t"
                "vsrl.vv v2, v0, v8\n\t"
                "vand.vx v5, v4, %[kmask1]\n\t"
                "vand.vx v3, v2, %[kmask2]\n\t"
                "vsll.vi v6, v5, 4\n\t"
                "vor.vv v7, v6, v3\n\t"
                "vsetivli zero, 16, e8, m1\n\t"
                "vsub.vx v0, v7, %[c]\n\t"
                "vse8.v v0, (%[scale])"
                : [tmp] "=&r"(tmp)
                : [sh] "r"(0x0000000400000004), [s6b] "r"(x[i].scales), [c] "r"(32), [scale] "r"(scale), [kmask1] "r"(kmask1), [kmask2] "r"(kmask2)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

            uint8_t m = 1;
            int isum = 0;
            for (int j = 0; j < QK_K; j += 128) {
                __asm__ __volatile__(
                    "vsetvli zero, %[vl32], e8, m2, ta, mu\n\t"
                    "vle8.v v8, (%[q3])\n\t"
                    "vsrl.vi v10, v8, 2\n\t"
                    "vsrl.vi v12, v8, 4\n\t"
                    "vsrl.vi v14, v8, 6\n\t"
                    "vand.vi v8, v8, 3\n\t"
                    "vand.vi v10, v10, 3\n\t"
                    "vand.vi v12, v12, 3\n\t"
                    "vle8.v v2, (%[qh])\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v8, v8, -4, v0.t\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v10, v10, -4, v0.t\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v12, v12, -4, v0.t\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v14, v14, -4, v0.t\n\t"
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vle8.v v0, (%[q8])\n\t"
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
                    "vsext.vf4 v12, v15\n\t"
                    "vmul.vv v10, v10, v12\n\t"
                    "vredsum.vs v0, v10, v0\n\t"
                    "vmv.x.s %[tmp], v0\n\t"
                    "add %[isum], %[isum], %[tmp]"
                    : [tmp] "=&r"(tmp), [m] "+&r"(m), [isum] "+&r"(isum)
                    : [vl128] "r"(128), [vl64] "r"(64), [vl32] "r"(32), [q3] "r"(q3), [qh] "r"(qh), [scale] "r"(scale), [q8] "r"(q8)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                q3 += 32;
                q8 += 128;
                scale += 8;
            }

            const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
            sumf += d * isum;
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0x3);
    const vector signed char lowMask1 = vec_splats((int8_t)0xf);
    const vector signed char lowMask2 = vec_splats((int8_t)0x30);
    const vector int v0 = vec_splats((int32_t)0);
    const vector signed char v1 = vec_splats((signed char)0x1);
    const vector unsigned char v2 = vec_splats((unsigned char)0x2);
    const vector unsigned char v3 = vec_splats((unsigned char)0x3);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);
    const vector unsigned char v6 = vec_splats((unsigned char)0x6);
    const vector signed char off = vec_splats((signed char)0x20);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(MLLM_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        UNUSED(kmask1);
        UNUSED(kmask2);

        vector signed char u0 = (vector signed char)vec_xl_len(x[i].scales, 8);
        vector signed char u1 = vec_and(u0, lowMask1);
        vector signed char u2 = (vector signed char)vec_xl_len(x[i].scales + 8, 4);
        vector signed char u3 = (vector signed char)vec_mergeh((vector signed int)u2, (vector signed int)vec_sr(u2, v2));
        vector signed char u30 = vec_sl(vec_and(u3, lowMask), v4);
        vector signed char u31 = vec_and(u3, lowMask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        vector signed char vscales = (vector signed char)vec_mergeh((vector signed long long)u1, (vector signed long long)u2);
        vector signed char qxhs0 = (vector signed char)vec_xl(0, x[i].hmask);
        vector signed char qxhs1 = (vector signed char)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;
        vector signed int vsumi4 = v0;
        vector signed int vsumi5 = v0;
        vector signed int vsumi6 = v0;
        vector signed int vsumi7 = v0;

        const uint8_t *__restrict q3 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K / 128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl(0, q3);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q3);
            q3 += 32;

            // the low 2 bits
            vector signed char qxs00 = vec_and(qxs0, lowMask);
            vector signed char qxs01 = vec_and(vec_sr(qxs0, v2), lowMask);
            vector signed char qxs02 = vec_and(vec_sr(qxs0, v4), lowMask);
            vector signed char qxs03 = vec_and(vec_sr(qxs0, v6), lowMask);
            vector signed char qxs10 = vec_and(qxs1, lowMask);
            vector signed char qxs11 = vec_and(vec_sr(qxs1, v2), lowMask);
            vector signed char qxs12 = vec_and(vec_sr(qxs1, v4), lowMask);
            vector signed char qxs13 = vec_and(vec_sr(qxs1, v6), lowMask);

            // the 3rd bit
            vector signed char qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            vector signed char qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (vector unsigned char)v1)), v2);
            vector signed char qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            vector signed char qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            vector signed char qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            vector signed char qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (vector unsigned char)v1)), v2);
            vector signed char qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            vector signed char qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            vector signed char q3x00 = vec_sub(qxs00, qxh00);
            vector signed char q3x01 = vec_sub(qxs01, qxh01);
            vector signed char q3x02 = vec_sub(qxs02, qxh02);
            vector signed char q3x03 = vec_sub(qxs03, qxh03);
            vector signed char q3x10 = vec_sub(qxs10, qxh10);
            vector signed char q3x11 = vec_sub(qxs11, qxh11);
            vector signed char q3x12 = vec_sub(qxs12, qxh12);
            vector signed char q3x13 = vec_sub(qxs13, qxh13);

            vector signed char q8y00 = vec_xl(0, q8);
            vector signed char q8y10 = vec_xl(16, q8);
            vector signed char q8y01 = vec_xl(32, q8);
            vector signed char q8y11 = vec_xl(48, q8);
            vector signed char q8y02 = vec_xl(64, q8);
            vector signed char q8y12 = vec_xl(80, q8);
            vector signed char q8y03 = vec_xl(96, q8);
            vector signed char q8y13 = vec_xl(112, q8);
            q8 += 128;

            vector signed short vscales_h = vec_unpackh(vscales);
            vector signed short vs0 = vec_splat(vscales_h, 0);
            vector signed short vs1 = vec_splat(vscales_h, 1);
            vector signed short vs2 = vec_splat(vscales_h, 2);
            vector signed short vs3 = vec_splat(vscales_h, 3);
            vector signed short vs4 = vec_splat(vscales_h, 4);
            vector signed short vs5 = vec_splat(vscales_h, 5);
            vector signed short vs6 = vec_splat(vscales_h, 6);
            vector signed short vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            vector signed short qv00 = vec_add(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            vector signed short qv01 = vec_add(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            vector signed short qv02 = vec_add(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            vector signed short qv03 = vec_add(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            vector signed short qv10 = vec_add(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            vector signed short qv11 = vec_add(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            vector signed short qv12 = vec_add(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            vector signed short qv13 = vec_add(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
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

    const __m128i m32 = __lsx_vreplgr2vr_b(32);

    __m256 acc = (__m256)__lasx_xvldi(0);

    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const uint8_t *__restrict q3 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        // Set up scales
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = lsx_set_w(
            ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
            ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
            (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
            (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = __lsx_vsub_b(scales128, m32);

        const v16i8 shuffle_mask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        const __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_mask));

        // high bit
        const __m256i hbits = __lasx_xvld((const __m256i *)x[i].hmask, 0);

        // integer accumulator
        __m256i sumi = __lasx_xvldi(0);

        for (int j = 0; j < QK_K / 128; ++j) {
            // load low 2 bits
            const __m256i q3bits = __lasx_xvld((const __m256i *)q3, 0);
            q3 += 32;

            // prepare low and high bits
            const __m256i q3l_0 = __lasx_xvandi_b(q3bits, 3);
            const __m256i q3l_1 = __lasx_xvandi_b(__lasx_xvsrli_b(q3bits, 2), 3);
            const __m256i q3l_2 = __lasx_xvandi_b(__lasx_xvsrli_b(q3bits, 4), 3);
            const __m256i q3l_3 = __lasx_xvsrli_b(q3bits, 6);
            const __m256i q3h_0 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 0), 0), 2);
            const __m256i q3h_1 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 1), 0), 2);
            const __m256i q3h_2 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 2), 0), 2);
            const __m256i q3h_3 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 3), 0), 2);
            const __m256i q3_0 = __lasx_xvor_v(q3h_0, q3l_0);
            const __m256i q3_1 = __lasx_xvor_v(q3h_1, q3l_1);
            const __m256i q3_2 = __lasx_xvor_v(q3h_2, q3l_2);
            const __m256i q3_3 = __lasx_xvor_v(q3h_3, q3l_3);

            // load Q8 quants
            const __m256i q8_0 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;
            const __m256i q8_3 = __lasx_xvld((const __m256i *)q8, 0);
            q8 += 32;

            __m256i p16_0 = lasx_madd_h_b(q8_0, q3_0);
            __m256i p16_1 = lasx_madd_h_b(q8_1, q3_1);
            __m256i p16_2 = lasx_madd_h_b(q8_2, q3_2);
            __m256i p16_3 = lasx_madd_h_b(q8_3, q3_3);

            // multiply with scales
            p16_0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p16_0);
            p16_1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p16_1);
            p16_2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p16_2);
            p16_3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p16_3);

            // accumulate
            p16_0 = __lasx_xvadd_w(p16_0, p16_1);
            p16_2 = __lasx_xvadd_w(p16_2, p16_3);
            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_2));
        }
        // multiply with block scale and accumulate
        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__VXE__) || defined(__VXE2__)
    uint32_t aux[3];
    uint32_t utmp[4];

    const int32x4_t v_z = vec_splat_s32(0);
    const uint8x16_t v_3m = vec_splat_u8(0x03);

    const uint8x16_t v_0c = vec_splat_u8(1);
    const uint8x16_t v_1c = vec_sl(v_0c, 1);
    const uint8x16_t v_2c = vec_sl(v_0c, 2);
    const uint8x16_t v_3c = vec_sl(v_0c, 3);

    uint8x16_t q3h[4];
    uint8x16_t q3b[2];
    int8x16_t q3bytes[4];
    int8x16_t q8bytes[4];
    uint8x16_t qhbits[2];

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *restrict x0l = x[i].qs;
        const uint8_t *restrict x0h = x[i].hmask;
        const int8_t *restrict y0 = y[i].qs;

        qhbits[0] = vec_xl(0, x0h);
        qhbits[1] = vec_xl(16, x0h);

        int32_t isum = 0;

        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t *scale = (int8_t *)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= 32;

        for (int j = 0; j < QK_K / 128; ++j) {
            int32x4_t isum0, isum1, isum2, isum3;

            q3b[0] = vec_xl(0, x0l);
            q3b[1] = vec_xl(16, x0l);
            x0l += 32;

            q8bytes[0] = vec_xl(0, y0);
            q8bytes[1] = vec_xl(16, y0);
            q8bytes[2] = vec_xl(32, y0);
            q8bytes[3] = vec_xl(48, y0);
            q8bytes[4] = vec_xl(64, y0);
            q8bytes[5] = vec_xl(80, y0);
            q8bytes[6] = vec_xl(96, y0);
            q8bytes[7] = vec_xl(112, y0);
            y0 += 128;

            q3h[0] = vec_sl(vec_andc(v_0c, qhbits[0]), 2);
            q3h[1] = vec_sl(vec_andc(v_0c, qhbits[1]), 2);
            q3h[2] = vec_sl(vec_andc(v_1c, qhbits[0]), 1);
            q3h[3] = vec_sl(vec_andc(v_1c, qhbits[1]), 1);

            q3bytes[0] = vec_sub((int8x16_t)vec_and(q3b[0], v_3m), (int8x16_t)q3h[0]);
            q3bytes[1] = vec_sub((int8x16_t)vec_and(q3b[1], v_3m), (int8x16_t)q3h[1]);
            q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 2), v_3m), (int8x16_t)q3h[2]);
            q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 2), v_3m), (int8x16_t)q3h[3]);

            isum0 = mllm_vec_dot(v_z, q3bytes[0], q8bytes[0]);
            isum1 = mllm_vec_dot(v_z, q3bytes[1], q8bytes[1]);
            isum2 = mllm_vec_dot(v_z, q3bytes[2], q8bytes[2]);
            isum3 = mllm_vec_dot(v_z, q3bytes[3], q8bytes[3]);

            isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
            isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
            isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
            isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

            scale += 4;

            q3h[0] = vec_andc(v_2c, qhbits[0]);
            q3h[1] = vec_andc(v_2c, qhbits[1]);
            q3h[2] = vec_sr(vec_andc(v_3c, qhbits[0]), 1);
            q3h[3] = vec_sr(vec_andc(v_3c, qhbits[1]), 1);

            q3bytes[0] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 4), v_3m), (int8x16_t)q3h[0]);
            q3bytes[1] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 4), v_3m), (int8x16_t)q3h[1]);
            q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 6), v_3m), (int8x16_t)q3h[2]);
            q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 6), v_3m), (int8x16_t)q3h[3]);

            isum0 = mllm_vec_dot(v_z, q3bytes[0], q8bytes[4]);
            isum1 = mllm_vec_dot(v_z, q3bytes[1], q8bytes[5]);
            isum2 = mllm_vec_dot(v_z, q3bytes[2], q8bytes[6]);
            isum3 = mllm_vec_dot(v_z, q3bytes[3], q8bytes[7]);

            isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
            isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
            isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
            isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

            scale += 4;

            if (j == 0) {
                qhbits[0] = vec_sr(qhbits[0], 4);
                qhbits[1] = vec_sr(qhbits[1], 4);
            }
        }

        sum += d * isum;
    }

    *s = sum;
#else
    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    uint32_t auxs[4];
    const int8_t *scales = (const int8_t *)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *__restrict q3 = x[i].qs;
        const uint8_t *__restrict hm = x[i].hmask;
        const int8_t *__restrict q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *__restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K / 16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8;
            a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8;
            a += 8;
        }
        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif
}