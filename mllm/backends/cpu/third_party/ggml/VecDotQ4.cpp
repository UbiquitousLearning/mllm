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

#include "VecDotQ4.hpp"
#include "ComputeUtils.hpp"

#if QK_K == 256
void vec_dot_q4_K_q8_K(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    assert(n % QK_K == 0);

    const block_q4_K *__restrict x = (block_q4_K *)vx;
    const block_q8_K *__restrict y = (block_q8_K *)vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ifdef __ARM_FEATURE_SVE
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, K_SCALE_SIZE);

        uint32x2_t mins8 = {0};
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t *scales = (const uint8_t *)utmp;

        const uint8_t *__restrict q4 = (const uint8_t *)x[i].qs;
        const int8_t *__restrict q8 = (const int8_t *)y[i].qs;

        const int vector_length = mllm_cpu_get_sve_cnt() * 8;
        const svuint8_t m4b = svdup_n_u8(0xf);
        const svint32_t mzero = svdup_n_s32(0);
        svint32_t sumi1 = svdup_n_s32(0);
        svint32_t sumi1_1 = svdup_n_s32(0);
        svint32_t sumi1_2 = svdup_n_s32(0);
        svint32_t sumi2 = svdup_n_s32(0);
        svint32_t sumi2_1 = svdup_n_s32(0);
        svint32_t sumi2_2 = svdup_n_s32(0);
        switch (vector_length) {
        case 128: {
            for (int j = 0; j < QK_K / 64; ++j) {
                svint8_t q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4), m4b));
                svint8_t q8bytes = svld1_s8(svptrue_b8(), q8);
                q8 += 16;
                sumi1_1 = svmla_n_s32_x(svptrue_b32(), sumi1_1, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 0]);
                q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4 + 16), m4b));
                q8bytes = svld1_s8(svptrue_b8(), q8);
                q8 += 16;
                sumi1_2 = svmla_n_s32_x(svptrue_b32(), sumi1_2, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 0]);

                q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4), 4));
                q8bytes = svld1_s8(svptrue_b8(), q8);
                q8 += 16;
                sumi2_1 = svmla_n_s32_x(svptrue_b32(), sumi2_1, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 1]);
                q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4 + 16), 4));
                q8bytes = svld1_s8(svptrue_b8(), q8);
                q8 += 16;
                sumi2_2 = svmla_n_s32_x(svptrue_b32(), sumi2_2, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 1]);
                q4 += 32;
            }
            sumi1 = svadd_s32_x(svptrue_b32(), sumi1_1, sumi1_2);
            sumi2 = svadd_s32_x(svptrue_b32(), sumi2_1, sumi2_2);
            sumf += d * (svaddv_s32(svptrue_b32(), svadd_s32_x(svptrue_b32(), sumi1, sumi2)));
        } break;
        case 256:
        case 512: {
            for (int j = 0; j < QK_K / 64; ++j) {
                const svuint8_t q4bits = svld1_u8(svptrue_pat_b8(SV_VL32), q4);
                q4 += 32;
                svint8_t q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_pat_b8(SV_VL32), q4bits, m4b));
                svint8_t q8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), q8);
                q8 += 32;
                sumi1 = svmla_n_s32_x(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 0]);

                q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q4bits, 4));
                q8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), q8);
                q8 += 32;
                sumi2 = svmla_n_s32_x(svptrue_pat_b32(SV_VL8), sumi2, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 1]);
            }
            sumf += d * (svaddv_s32(svptrue_pat_b32(SV_VL8), svadd_s32_x(svptrue_pat_b32(SV_VL8), sumi1, sumi2)));
        } break;
        default:
            assert(false && "Unsupported vector length");
            break;
        }
    }
    *s = sumf;
#elif defined __ARM_NEON

    const uint8x16_t m4b = vdupq_n_u8(0xf);
#ifdef __ARM_FEATURE_DOTPROD
    const int32x4_t mzero = vdupq_n_s32(0);
#endif

    int8x16x2_t q4bytes;
    int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        const uint32x2_t mins8 = {utmp[1] & kmask1, ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4)};
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t *scales = (const uint8_t *)utmp;

        const uint8_t *__restrict q4 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        // int32x4_t isum = mzero;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K / 64; ++j) {
            const uint8x16x2_t q4bits = vld1q_u8_x2(q4);
            q4 += 32;

#ifdef __ARM_FEATURE_DOTPROD
            q8bytes = vld1q_s8_x2(q8);
            q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));

            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2 * j + 0];

            q8bytes = vld1q_s8_x2(q8);
            q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2 * j + 1];
#else
            q8bytes = vld1q_s8_x2(q8);
            q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));
            const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                                           vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                                           vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            sumi1 += vaddvq_s16(vaddq_s16(p0, p1)) * scales[2 * j + 0];

            q8bytes = vld1q_s8_x2(q8);
            q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
            const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                                           vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                                           vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            sumi2 += vaddvq_s16(vaddq_s16(p2, p3)) * scales[2 * j + 1];

#endif
        }

        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t *__restrict q4 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#else
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[2];

    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *__restrict q4 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *__restrict a = aux8;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8;
            a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8;
            a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8;
            a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8;
            a += 8;
        }
        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = MLLM_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#else
void vec_dot_q4_K_q8_K(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    assert(n % QK_K == 0);

    const block_q4_K *__restrict x = (block_q4_K *)vx;
    const block_q8_K *__restrict y = (block_q8_K *)vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    const uint8x16_t m4b = vdupq_n_u8(0xf);

#ifdef __ARM_FEATURE_DOTPROD
    const int32x4_t mzero = vdupq_n_s32(0);
#endif

    float sumf = 0;

    int8x16x2_t q4bytes;
    int8x16x4_t q8bytes;

    float sum_mins = 0.f;

    uint16_t aux16[2];
    const uint8_t *__restrict scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {
        const uint8_t *__restrict q4 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        const uint16_t *__restrict a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        const int32_t summi = scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]);
        sum_mins += y[i].d * (float)x[i].d[1] * summi;

        const float d = y[i].d * (float)x[i].d[0];

        const uint8x16x2_t q4bits = vld1q_u8_x2(q4);

#ifdef __ARM_FEATURE_DOTPROD
        q8bytes = vld1q_s8_x4(q8);
        q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
        q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));

        const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
        const int32_t sumi1 = vaddvq_s32(p1) * scales[0];

        q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
        q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

        const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[2]), q4bytes.val[1], q8bytes.val[3]);
        const int32_t sumi2 = vaddvq_s32(p2) * scales[1];

#else
        q8bytes = vld1q_s8_x4(q8);
        q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
        q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));
        const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        int32_t sumi1 = vaddvq_s16(vaddq_s16(p0, p1)) * scales[0];

        q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
        q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[0]), vget_low_s8(q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[1]), vget_low_s8(q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[3])));
        int32_t sumi2 = vaddvq_s16(vaddq_s16(p2, p3)) * scales[1];

#endif
        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf - sum_mins;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    uint16_t aux16[2];
    const uint8_t *scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {
        const float d = MLLM_FP16_TO_FP32(x[i].d[0]) * y[i].d;
        const float m = MLLM_FP16_TO_FP32(x[i].d[1]) * y[i].d;
        const __m256 vd = _mm256_set1_ps(d);

        const uint16_t *a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        summs += m * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const uint8_t *__restrict q4 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;

        const __m256i q4bits = _mm256_loadu_si256((const __m256i *)q4);
        const __m256i q4l = _mm256_and_si256(q4bits, m4);
        const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        const __m256i q8l = _mm256_loadu_si256((const __m256i *)(q8 + 0));
        const __m256i q8h = _mm256_loadu_si256((const __m256i *)(q8 + 32));

        const __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
        const __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);

        const __m256i p32l = _mm256_madd_epi16(_mm256_set1_epi16(scales[0]), p16l);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32l), acc);

        const __m256i p32h = _mm256_madd_epi16(_mm256_set1_epi16(scales[1]), p16h);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32h), acc);
    }

    *s = hsum_float_8(acc) - summs;

#else

    uint8_t aux8[QK_K];
    int16_t aux16[16];
    float sums[8];
    memset(sums, 0, 8 * sizeof(float));

    uint16_t s16[2];
    const uint8_t *__restrict scales = (const uint8_t *)s16;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *__restrict q4 = x[i].qs;
        const int8_t *__restrict q8 = y[i].qs;
        uint8_t *__restrict a = aux8;
        for (int l = 0; l < 32; ++l) a[l + 0] = q4[l] & 0xF;
        for (int l = 0; l < 32; ++l) a[l + 32] = q4[l] >> 4;

        const uint16_t *__restrict b = (const uint16_t *)x[i].scales;
        s16[0] = b[0] & 0x0f0f;
        s16[1] = (b[0] >> 4) & 0x0f0f;

        sumf -= y[i].d * MLLM_FP16_TO_FP32(x[i].d[1]) * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d[0]);

        for (int j = 0; j < QK_K / 32; ++j) {
            for (int l = 0; l < 16; ++l) aux16[l] = q8[l] * a[l];
            q8 += 16;
            a += 16;
            for (int l = 0; l < 16; ++l) aux16[l] += q8[l] * a[l];
            q8 += 16;
            a += 16;
            const float dl = d * scales[j];
            for (int l = 0; l < 8; ++l) sums[l] += dl * (aux16[l] + aux16[l + 8]);
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#endif
#ifdef __AVX2__
static void vec_dot_q4_0_q8_0_avx(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);

    const block_q4_0 *__restrict x = (block_q4_0 *)vx;
    const block_q8_0 *__restrict y = (block_q8_0 *)vy;
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(MLLM_FP16_TO_FP32(x[i].d) * MLLM_FP16_TO_FP32(y[i].d));

        __m256i bx = bytes_from_nibbles_32(x[i].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8(8);
        bx = _mm256_sub_epi8(bx, off);

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    *s = hsum_float_8(acc);
}
#endif
#ifdef __ARM_NEON
static void vec_dot_q4_0_q8_0_arm(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);

    const block_q4_0 *__restrict x = (block_q4_0 *)vx;
    const block_q8_0 *__restrict y = (block_q8_0 *)vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    {
        size_t bs = 0;
        size_t bx = 0;
        size_t by = 0;
        const block_q4_0 *__restrict vx0 = (const block_q4_0 *)vx;
        const block_q4_0 *__restrict vx1 = (const block_q4_0 *)((const uint8_t *)vx + bx);
        const block_q8_0 *__restrict vy0 = (const block_q8_0 *)vy;
        const block_q8_0 *__restrict vy1 = (const block_q8_0 *)((const uint8_t *)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_0 *__restrict b_x0 = &vx0[i];
            const block_q4_0 *__restrict b_x1 = &vx1[i];
            const block_q8_0 *__restrict b_y0 = &vy0[i];
            const block_q8_0 *__restrict b_y1 = &vy1[i];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
            const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
            const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
            const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                MLLM_FP16_TO_FP32(b_x0->d) * MLLM_FP16_TO_FP32(b_y0->d),
                MLLM_FP16_TO_FP32(b_x0->d) * MLLM_FP16_TO_FP32(b_y1->d),
                MLLM_FP16_TO_FP32(b_x1->d) * MLLM_FP16_TO_FP32(b_y0->d),
                MLLM_FP16_TO_FP32(b_x1->d) * MLLM_FP16_TO_FP32(b_y1->d)};
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0, (vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)), l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s, vget_low_f32(sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    float32x4_t sumv0 = vdupq_n_f32(0.0F);
    float32x4_t sumv1 = vdupq_n_f32(0.0F);

    assert(nb % 2 == 0); // TODO: handle odd nb
    for (int i = 0; i < nb; i += 2) {
        const block_q4_0 *__restrict x0 = &x[i + 0];
        const block_q4_0 *__restrict x1 = &x[i + 1];
        const block_q8_0 *__restrict y0 = &y[i + 0];
        const block_q8_0 *__restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), MLLM_FP16_TO_FP32(x0->d) * MLLM_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), MLLM_FP16_TO_FP32(x1->d) * MLLM_FP16_TO_FP32(y1->d));
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8(v0_0ls), vget_low_s8(v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8(v0_0hs), vget_low_s8(v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8(v0_1ls), vget_low_s8(v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8(v0_1hs), vget_low_s8(v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), MLLM_FP16_TO_FP32(x0->d) * MLLM_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), MLLM_FP16_TO_FP32(x1->d) * MLLM_FP16_TO_FP32(y1->d));
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
}
#endif

void vec_dot_q4_0_q8_0(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
#ifdef __AVX2__
    vec_dot_q4_0_q8_0_avx(n, s, vx, vy);
#elif defined(__ARM_NEON)
    vec_dot_q4_0_q8_0_arm(n, s, vx, vy);
#endif
}