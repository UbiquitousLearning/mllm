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

#include "VecDotQ6.hpp"
#include "ComputeUtils.hpp"

#if QK_K == 256
void vec_dot_q6_K_q8_K(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    assert(n % QK_K == 0);

    const block_q6_K *__restrict x = (block_q6_K *)vx;
    const block_q8_K *__restrict y = (block_q8_K *)vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    float sum = 0;

    const uint8x16_t m4b = vdupq_n_u8(0xF);
#if defined(__ARM_FEATURE_DOTPROD)
    const int32x4_t vzero = vdupq_n_s32(0);
#endif
    // const int8x16_t  m32s = vdupq_n_s8(32);

    const uint8x16_t mone = vdupq_n_u8(3);

    int8x16x4_t q6bytes;
    uint8x16x4_t q6h;

    for (int i = 0; i < nb; ++i) {
        const float d_all = MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q6 = x[i].ql;
        const uint8_t *__restrict qh = x[i].qh;
        const int8_t *__restrict q8 = y[i].qs;

        const int8_t *__restrict scale = x[i].scales;

        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int8x16_t scales = vld1q_s8(scale);
        const int16x8x2_t q6scales = {vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))};

        const int32x4_t prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]), vget_low_s16(q6scales.val[0])),
                                                   vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                                         vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]), vget_low_s16(q6scales.val[1])),
                                                   vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
        int32_t isum_mins = vaddvq_s32(prod);

        int32_t isum = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            uint8x16x2_t qhbits = vld1q_u8_x2(qh);
            qh += 32;
            uint8x16x4_t q6bits = vld1q_u8_x4(q6);
            q6 += 64;
            int8x16x4_t q8bytes = vld1q_s8_x4(q8);
            q8 += 64;

            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 2);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
            // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
            // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
            // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

            isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
            scale += 4;

#else

            int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                                     vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                                     vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];
            scale += 2;

            int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                                     vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
            int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                                     vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
            isum += vaddvq_s16(p2) * scale[0] + vaddvq_s16(p3) * scale[1];
            scale += 2;
#endif

            q8bytes = vld1q_s8_x4(q8);
            q8 += 64;

            shifted = vshrq_n_u8(qhbits.val[0], 4);
            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[0], 6);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 6);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
            // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
            // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
            // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

            isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
            scale += 4;

            // for (int l = 0; l < 4; ++l) {
            //     const int32x4_t p = vdotq_s32(vzero, q6bytes.val[l], q8bytes.val[l]);
            //     isum += vaddvq_s32(p) * *scale++;
            // }
#else
            p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                           vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                           vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];
            scale += 2;

            p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                           vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
            p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                           vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
            isum += vaddvq_s16(p2) * scale[0] + vaddvq_s16(p3) * scale[1];
            scale += 2;
#endif
        }
        // sum += isum * d_all * y[i].d;
        sum += d_all * y[i].d * (isum - 32 * isum_mins);
    }
    *s = sum;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q4 = x[i].ql;
        const uint8_t *__restrict qh = x[i].qh;
        const int8_t *__restrict q8 = y[i].qs;

        const __m128i scales = _mm_loadu_si128((const __m128i *)x[i].scales);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
            const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
            const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

#if defined(_MSC_VER) || defined(__MINGW32__)
    float arr[8];
    _mm256_storeu_ps(arr, acc);

    // for(float i : arr) {
    //     printf("%f ", i);
    // }
    // printf("\n");
#endif

    *s = hsum_float_8(acc);

#elif defined __AVX__
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m32s = _mm_set1_epi8(32);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q4 = x[i].ql;
        const uint8_t *__restrict qh = x[i].qh;
        const int8_t *__restrict q8 = y[i].qs;

        const __m128i scales = _mm_loadu_si128((const __m128i *)x[i].scales);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i *)qh);
            qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i *)qh);
            qh += 16;

            const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
            const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
            const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 2), m3), 4);
            const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 2), m3), 4);
            const __m128i q4h_4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 4), m3), 4);
            const __m128i q4h_5 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 4), m3), 4);
            const __m128i q4h_6 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 6), m3), 4);
            const __m128i q4h_7 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 6), m3), 4);

            const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i *)q4);
            q4 += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i *)q4);
            q4 += 16;
            const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i *)q4);
            q4 += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i *)q4);
            q4 += 16;

            const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m4), q4h_0);
            const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m4), q4h_1);
            const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m4), q4h_2);
            const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m4), q4h_3);
            const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m4), q4h_4);
            const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m4), q4h_5);
            const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m4), q4h_6);
            const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m4), q4h_7);

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

            __m128i q8s_0 = _mm_maddubs_epi16(m32s, q8_0);
            __m128i q8s_1 = _mm_maddubs_epi16(m32s, q8_1);
            __m128i q8s_2 = _mm_maddubs_epi16(m32s, q8_2);
            __m128i q8s_3 = _mm_maddubs_epi16(m32s, q8_3);
            __m128i q8s_4 = _mm_maddubs_epi16(m32s, q8_4);
            __m128i q8s_5 = _mm_maddubs_epi16(m32s, q8_5);
            __m128i q8s_6 = _mm_maddubs_epi16(m32s, q8_6);
            __m128i q8s_7 = _mm_maddubs_epi16(m32s, q8_7);

            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            p16_0 = _mm_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm_sub_epi16(p16_3, q8s_3);
            p16_4 = _mm_sub_epi16(p16_4, q8s_4);
            p16_5 = _mm_sub_epi16(p16_5, q8s_5);
            p16_6 = _mm_sub_epi16(p16_6, q8s_6);
            p16_7 = _mm_sub_epi16(p16_7, q8s_7);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi8(shuffle, m2);
            const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi8(shuffle, m2);
            const __m128i scale_2 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi8(shuffle, m2);
            const __m128i scale_3 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi8(shuffle, m2);

            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_2, scale_2)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_3, scale_3)), p16_7);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));
        }

        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#else

    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *__restrict q4 = x[i].ql;
        const uint8_t *__restrict qh = x[i].qh;
        const int8_t *__restrict q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *__restrict a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            int scale = x[i].scales[is++];
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
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

#else

void vec_dot_q6_K_q8_K(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy) {
    assert(n % QK_K == 0);

    const block_q6_K *__restrict x = (block_q6_K *)vx;
    const block_q8_K *__restrict y = (block_q8_K *)vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    float sum = 0;

    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int8x16_t m32s = vdupq_n_s8(32);
#if defined(__ARM_FEATURE_DOTPROD)
    const int32x4_t vzero = vdupq_n_s32(0);
#endif

    const uint8x16_t mone = vdupq_n_u8(3);

    int8x16x4_t q6bytes;
    uint8x16x4_t q6h;

    for (int i = 0; i < nb; ++i) {
        const float d_all = (float)x[i].d;

        const uint8_t *__restrict q6 = x[i].ql;
        const uint8_t *__restrict qh = x[i].qh;
        const int8_t *__restrict q8 = y[i].qs;

        const int8_t *__restrict scale = x[i].scales;

        int32_t isum = 0;

        uint8x16_t qhbits = vld1q_u8(qh);
        uint8x16x2_t q6bits = vld1q_u8_x2(q6);
        int8x16x4_t q8bytes = vld1q_s8_x4(q8);

        q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits), 4);
        uint8x16_t shifted = vshrq_n_u8(qhbits, 2);
        q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits, 4);
        q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits, 6);
        q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

        q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
        q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
        q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[2])), m32s);
        q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[3])), m32s);

#if defined(__ARM_FEATURE_DOTPROD)

        isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
#else

        int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                                 vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                                 vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];

        int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                                 vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                                 vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        isum += vaddvq_s16(p2) * scale[2] + vaddvq_s16(p3) * scale[3];
#endif

        sum += isum * d_all * y[i].d;
    }
    *s = sum;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q4 = x[i].ql;
        const uint8_t *__restrict qh = x[i].qh;
        const int8_t *__restrict q8 = y[i].qs;

        const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
        const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
        const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
        const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

        __m256i sumi = _mm256_setzero_si256();

        const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
        const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

        const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
        const __m128i q4bitsH = _mm_loadu_si128((const __m128i *)qh);

        const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 2), q4bitsH), m2), 4);
        const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 6), _mm_srli_epi16(q4bitsH, 4)), m2), 4);

        const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
        const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_1);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)(q8 + 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)(q8 + 32));

        __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
        __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);

        __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
        __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);

        p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm256_sub_epi16(p16_1, q8s_1);

        p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
        p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);

        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(3);
    const __m128i m32s = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

        const uint8_t *__restrict q4 = x[i].ql;
        const uint8_t *__restrict qh = x[i].qh;
        const int8_t *__restrict q8 = y[i].qs;

        const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
        const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
        const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
        const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
        const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

        const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
        const __m128i q4bitsH = _mm_loadu_si128((const __m128i *)qh);

        const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH, m2), 4);
        const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 2), m2), 4);
        const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 4), m2), 4);
        const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 6), m2), 4);

        const __m128i q4_0 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 0), m4), q4h_0);
        const __m128i q4_1 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 1), m4), q4h_1);
        const __m128i q4_2 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 0), 4), m4), q4h_2);
        const __m128i q4_3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 1), 4), m4), q4h_3);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)(q8 + 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)(q8 + 32));

        __m128i q8s_0 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 0));
        __m128i q8s_1 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 1));
        __m128i q8s_2 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 0));
        __m128i q8s_3 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 1));

        __m128i p16_0 = _mm_maddubs_epi16(q4_0, _mm256_extractf128_si256(q8_0, 0));
        __m128i p16_1 = _mm_maddubs_epi16(q4_1, _mm256_extractf128_si256(q8_0, 1));
        __m128i p16_2 = _mm_maddubs_epi16(q4_2, _mm256_extractf128_si256(q8_1, 0));
        __m128i p16_3 = _mm_maddubs_epi16(q4_3, _mm256_extractf128_si256(q8_1, 1));

        p16_0 = _mm_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm_sub_epi16(p16_1, q8s_1);
        p16_2 = _mm_sub_epi16(p16_2, q8s_2);
        p16_3 = _mm_sub_epi16(p16_3, q8s_3);

        p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
        p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
        p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
        p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);

        sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
        sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi_1, sumi_0))), acc);
    }

    *s = hsum_float_8(acc);

#else

    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *__restrict q4 = x[i].ql;
        const uint8_t *__restrict qh = x[i].qh;
        const int8_t *__restrict q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *__restrict a = aux8;
        for (int l = 0; l < 16; ++l) {
            a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            a[l + 16] = (int8_t)((q4[l + 16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            a[l + 32] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            a[l + 48] = (int8_t)((q4[l + 16] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        }
        int is = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            int scale = x[i].scales[is++];
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
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#endif