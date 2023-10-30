//
// Created by ey on 23-10-24.
//

#include "Matmul.hpp"
#include <chrono>
#ifdef __ARM_NEON
//#include "../neon/Neon.hpp"
#include <arm_neon.h>
#endif
#define F32_BLOCK 16

#ifdef __AVX2__
//  COPY FROM MLLM
#define MLLM_F32_STEP 32
#define MLLM_F32_EPR 8
#define MLLM_F32_ARR (MLLM_F32_STEP / MLLM_F32_EPR)
#define MLLM_F32x8 __m256
#define MLLM_F32x8_ZERO _mm256_setzero_ps()
#define MLLM_F32x8_SET1(x) _mm256_set1_ps(x)
#define MLLM_F32x8_LOAD _mm256_loadu_ps
#define MLLM_F32x8_STORE _mm256_storeu_ps
#if defined(__FMA__)
#define MLLM_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
#define MLLM_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define MLLM_F32x8_ADD _mm256_add_ps
#define MLLM_F32x8_MUL _mm256_mul_ps
#define MLLM_F32x8_REDUCE(res, x)                                     \
    {                                                                 \
        int offset = MLLM_F32_ARR >> 1;                               \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset + i]);                \
        }                                                             \
        offset >>= 1;                                                 \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset + i]);                \
        }                                                             \
        offset >>= 1;                                                 \
        for (int i = 0; i < offset; ++i) {                            \
            x[i] = _mm256_add_ps(x[i], x[offset + i]);                \
        }                                                             \
        const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                     _mm256_extractf128_ps(x[0], 1)); \
        const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
        res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));                     \
    }
#define MLLM_F32x8 __m256
#define MLLM_F32_VEC MLLM_F32x8
#define MLLM_F32_VEC_ZERO MLLM_F32x8_ZERO
#define MLLM_F32_VEC_SET1 MLLM_F32x8_SET1
#define MLLM_F32_VEC_LOAD MLLM_F32x8_LOAD
#define MLLM_F32_VEC_STORE MLLM_F32x8_STORE
#define MLLM_F32_VEC_FMA MLLM_F32x8_FMA
#define MLLM_F32_VEC_ADD MLLM_F32x8_ADD
#define MLLM_F32_VEC_MUL MLLM_F32x8_MUL
#define MLLM_F32_VEC_REDUCE MLLM_F32x8_REDUCE
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}
// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}
static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if __AVXVNNI__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}
// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#endif

#ifdef __ARM_NEON
#define COMPUTE_FP16_TO_FP32(x) ((float) (x))
#define COMPUTE_FP32_TO_FP16(x) (x)
#define FP16_TO_FP32(x) ((float) (x))
#define FP32_TO_FP16(x) (x)
#define F32_VEC float32x4_t
#define F32_STEP 16                // 16 elements per step
#define F32_REG 4                  // 4 elements per register
#define F32_ARR F32_STEP / F32_REG // Len of sum array
#define F32_VEC_REDUCE(res, x)                     \
    {                                              \
        int offset = F32_ARR >> 1;                 \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        offset >>= 1;                              \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        offset >>= 1;                              \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        res = vaddvq_f32(x[0]);                    \
    }
#endif

#ifdef __AVX2__
static void vec_dot_fp32_avx2(const int n, float *__restrict s, const float *__restrict x, const float *__restrict y) {
    float sumf = 0.0F;
    const int np = (n & ~(MLLM_F32_STEP - 1));

    MLLM_F32_VEC sum[MLLM_F32_ARR] = {MLLM_F32_VEC_ZERO};

    MLLM_F32_VEC ax[MLLM_F32_ARR];
    MLLM_F32_VEC ay[MLLM_F32_ARR];

    for (int i = 0; i < np; i += MLLM_F32_STEP) {
        for (int j = 0; j < MLLM_F32_ARR; j++) {
            ax[j] = MLLM_F32_VEC_LOAD(x + i + j * MLLM_F32_EPR);
            ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);

            sum[j] = MLLM_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    MLLM_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }

    *s = sumf;
}
#endif

#ifdef __ARM_NEON
static void vec_dot_f32_arm(const int n, float *__restrict s, const float *__restrict x, const float *__restrict y) {
    float sumf = 0.0F;
    const int np = (n & ~(4 - 1));

    F32_VEC sum[4] = {vdupq_n_f32(0.0F)};

    F32_VEC ax[F32_ARR];
    F32_VEC ay[F32_ARR];

    for (int i = 0; i < np; i += F32_STEP) {
        for (int j = 0; j < F32_ARR; j++) {
            ax[j] = vld1q_f32(x + i + j * F32_REG);
            ay[j] = vld1q_f32(y + i + j * F32_REG);
            sum[j] = vmlaq_f32(sum[j], ax[j], ay[j]);
            // sum[j] = vmlaq_lane_f32(sum[j], ax[j], ay[0],
        }
    }

    // reduce sum0..sum3 to sum0
    F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }
}
#endif

void vec_dot_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf) {
    float value = 0;
#ifdef __AVX2__
    vec_dot_fp32_avx2(hid_len, &value, src0->ptrAt<float>(batch, head, src0_inf, 0), src1->ptrAt<float>(batch, head, sec1_outf, 0));
#elif defined(__ARM_NEON)
    vec_dot_f32_arm(hid_len, &value, src0->ptrAt<float>(batch, head, src0_inf, 0), src1->ptrAt<float>(batch, head, sec1_outf, 0));
#else
    for (int k = 0; k < hid_len; k++) {
        value += src0->dataAt<float>({batch, head, src0_inf, k}) * src1->dataAt<float>({batch, head, sec1_outf, k});
    }
    std::cout << value << ", " << value << std::endl;
#endif
    if (support_bias) {
        value += bias->dataAt<float>({0, head, 0, sec1_outf});
    }
    dst->setDataAt<float>({batch, head, src0_inf, sec1_outf}, value);
}

#ifdef __AVX2__
static void vec_dot_q4_0_q8_0_avx(const int n, float * __restrict s, const void * __restrict vx, const void * __restrict vy) {
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
        const __m256 d = _mm256_set1_ps( MLLM_FP16_TO_FP32(x[i].d) * MLLM_FP16_TO_FP32(y[i].d));

        __m256i bx = bytes_from_nibbles_32(x[i].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        bx = _mm256_sub_epi8( bx, off );

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }
    *s = hsum_float_8(acc);
}
#endif


#ifdef __ARM_NEON
// COPY FROMN
static void vec_dot_q4_0_q8_0_arm(const int n, float * __restrict s, const void * __restrict vx, const void * __restrict vy) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);

    const block_q4_0 *__restrict x = (block_q4_0 *)vx;
    const block_q8_0 *__restrict y = (block_q8_0 *)vy;
    float32x4_t sumv0 = vdupq_n_f32(0.0F);
    float32x4_t sumv1 = vdupq_n_f32(0.0F);

    assert(nb % 2 == 0); // TODO: handle odd nb
    for (int i = 0; i < nb; i += 2) {
        const block_q4_0 *__restrict x0 = &x[i + 0];
        const block_q4_0 *__restrict x1 = &x[i + 1];
        const block_q8_0 *__restrict y0 = &y[i + 0];
        const block_q8_0 *__restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
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

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), MLLM_FP16_TO_FP32(x0->d)*MLLM_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), MLLM_FP16_TO_FP32(x1->d)*MLLM_FP16_TO_FP32(y1->d));
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), MLLM_FP16_TO_FP32(x0->d)*MLLM_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), MLLM_FP16_TO_FP32(x1->d)*MLLM_FP16_TO_FP32(y1->d));
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
}
#endif

void vec_dot_q4_0_q8_0(const void * __restrict src0, const void * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf) {
    float value = 0;
#ifdef __AVX2__
    vec_dot_q4_0_q8_0_avx(hid_len, &value, src1, src0);
#elif defined(__ARM_NEON)
    vec_dot_q4_0_q8_0_arm(hid_len, &value, src1, src0);
#endif
    if (support_bias) {
        value += bias->dataAt<float>({0, head, 0, sec1_outf});
    }
    dst->setDataAt<float>({batch, head, src0_inf, sec1_outf}, value);
}

inline void transpose_scalar_block(const float *A, float *B, const int lda, const int ldb, const int block_size = F32_BLOCK) {
    int i;
    int j = 0;
    // Cache Aware Transpose
    #pragma omp parallel for num_threads(8)
    for (i = 0; i < block_size; i++) {
        for (j = 0; j < block_size; j++) {
            B[j * ldb + i] = A[i * lda + j];
        }
    }
}
Tensor *tensor_trans(Tensor *src) {
    Tensor *dst = new Tensor();
    dst->setBackend(src->backend());
    dst->reshape({src->batch(), src->head(), src->dimension(), src->sequence()});
    dst->setDtype(src->dtype());
    dst->alloc();
    for (int b = 0; b < src->batch(); b++) {
        for (int h = 0; h < src->head(); h++) {
            int i = 0;
            int j = 0;
            if (std::min(src->sequence(), src->dimension()) > F32_BLOCK) {
                #pragma omp parallel for num_threads(8)
                for (i = 0; i < src->sequence(); i += F32_BLOCK) {
                    for (j = 0; j < src->dimension(); j += F32_BLOCK) {
                        transpose_scalar_block(src->ptrAt<float>(b, h, i, j), dst->ptrAt<float>(b, h, j, i), src->dimension(), src->sequence());
                    }
                }
                // for leftovers
                for (; i < src->sequence(); i++) {
                    for (; j < src->dimension(); j++) {
                        dst->setDataAt<float>({b, h, j, i}, src->dataAt<float>({b, h, i, j}));
                    }
                }
                continue;
            }
            for (int n = 0; n < src->sequence(); n++) {
                for (int m = 0; m < src->dimension(); m++) {
                    dst->setDataAt<float>({b, h, m, n}, src->dataAt<float>({b, h, n, m}));
                }
            }
        }
    }
    return dst;
}

#if QK_K == 256
void vec_dot_q4_K_q8_K(const int n, float * __restrict s, const void * __restrict vx, const void * __restrict vy) {
    assert(n % QK_K == 0);

    const block_q4_K * __restrict x = (block_q4_K*)vx;
    const block_q8_K * __restrict y = (block_q8_K *)vy;

    const int nb = n / QK_K;

    static const uint32_t Kmask1 = 0x3f3f3f3f;
    static const uint32_t Kmask2 = 0x0f0f0f0f;
    static const uint32_t Kmask3 = 0x03030303;

    uint32_t utmp[4];

#ifdef __ARM_NEON

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

        const uint32x2_t mins8 = {utmp[1] & Kmask1, ((utmp[2] >> 4) & Kmask2) | (((utmp[1] >> 6) & Kmask3) << 4)};
        utmp[1] = (utmp[2] & Kmask2) | (((utmp[0] >> 6) & Kmask3) << 4);
        utmp[0] &= Kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        //int32x4_t isum = mzero;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const uint8x16x2_t q4bits = vld1q_u8_x2(q4); q4 += 32;

#ifdef __ARM_FEATURE_DOTPROD
            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
#else
            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));
            const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                           vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                           vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            sumi1 += vaddvq_s16(vaddq_s16(p0, p1)) * scales[2*j+0];

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
            const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                           vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                           vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            sumi2 += vaddvq_s16(vaddq_s16(p2, p3)) * scales[2*j+1];

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
        utmp[3] = ((utmp[2] >> 4) & Kmask2) | (((utmp[1] >> 6) & Kmask3) << 4);
        const uint32_t uaux = utmp[1] & Kmask1;
        utmp[1] = (utmp[2] & Kmask2) | (((utmp[0] >> 6) & Kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= Kmask1;

        const uint8_t * __restrict q4 = x[i].qs;
        const int8_t  * __restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
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
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & Kmask2) | (((utmp[1] >> 6) & Kmask3) << 4);
        const uint32_t uaux = utmp[1] & Kmask1;
        utmp[1] = (utmp[2] & Kmask2) | (((utmp[0] >> 6) & Kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= Kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
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
void vec_dot_q4_K_q8_K(const int n, float * __restrict s, const void * __restrict vx, const void * __restrict vy) {
    assert(n % QK_K == 0);

    const block_q4_K * __restrict x = (block_q4_K *)vx;
    const block_q8_K * __restrict y = (block_q8_K *)vy;

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
    const uint8_t * restrict scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint16_t * restrict a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        const int32_t summi = scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]);
        sum_mins += y[i].d * (float)x[i].d[1] * summi;

        const float d = y[i].d * (float)x[i].d[0];

        const uint8x16x2_t q4bits = vld1q_u8_x2(q4);

#ifdef __ARM_FEATURE_DOTPROD
        q8bytes = vld1q_s8_x4(q8);
        q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
        q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

        const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
        const int32_t sumi1 = vaddvq_s32(p1) * scales[0];

        q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
        q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

        const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[2]), q4bytes.val[1], q8bytes.val[3]);
        const int32_t sumi2 = vaddvq_s32(p2) * scales[1];

#else
        q8bytes = vld1q_s8_x4(q8);
        q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
        q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));
        const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        int32_t sumi1 = vaddvq_s16(vaddq_s16(p0, p1)) * scales[0];

        q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
        q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[3])),
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
    const uint8_t * scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = MLLM_FP16_TO_FP32(x[i].d[0]) * y[i].d;
        const float m = MLLM_FP16_TO_FP32(x[i].d[1]) * y[i].d;
        const __m256 vd = _mm256_set1_ps(d);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        summs += m * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const uint8_t * __restrict q4 = x[i].qs;
        const int8_t  * __restrict q8 = y[i].qs;

        const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
        const __m256i q4l = _mm256_and_si256(q4bits, m4);
        const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        const __m256i q8l = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8+32));

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
    float   sums [8];
    memset(sums, 0, 8*sizeof(float));

    uint16_t s16[2];
    const uint8_t * restrict scales = (const uint8_t *)s16;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        uint8_t * restrict a = aux8;
        for (int l = 0; l < 32; ++l) a[l+ 0] = q4[l] & 0xF;
        for (int l = 0; l < 32; ++l) a[l+32] = q4[l]  >> 4;

        const uint16_t * restrict b = (const uint16_t *)x[i].scales;
        s16[0] = b[0] & 0x0f0f;
        s16[1] = (b[0] >> 4) & 0x0f0f;

        sumf -= y[i].d * MLLM_FP16_TO_FP32(x[i].d[1]) * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d[0]);

        for (int j = 0; j < QK_K/32; ++j) {
            for (int l = 0; l < 16; ++l) aux16[l] = q8[l] * a[l];
            q8 += 16; a += 16;
            for (int l = 0; l < 16; ++l) aux16[l] += q8[l] * a[l];
            q8 += 16; a += 16;
            const float dl = d * scales[j];
            for (int l = 0; l < 8; ++l) sums[l] += dl * (aux16[l] + aux16[l+8]);
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#endif

void vec_dot_q4_K_q8_K(const void * __restrict src0, const void * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf) {
    float value = 0;

    vec_dot_q4_K_q8_K(hid_len, &value, src1, src0);

    if (support_bias) {
        value += bias->dataAt<float>({0, head, 0, sec1_outf});
    }
    dst->setDataAt<float>({batch, head, src0_inf, sec1_outf}, value);
}

ErrorCode mat_mul_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
    // INPUT: M.K
    // W:K,N
    // OUTPUT:M.N

    //    auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间

    int M = transpose0 ? src0->dimension() : src0->sequence();
    int K = transpose0 ? src0->sequence() : src0->dimension();
    int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = (transpose1 && !transpose0) ? src0 : (transpose0 && !transpose1) ? tensor_trans(src0) : src0;
    Tensor *src1_cal = (transpose1 && !transpose0) ? src1 : (!transpose0 && !transpose1) ? tensor_trans(src1) : src1;
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            #pragma omp parallel for num_threads(8)
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    vec_dot_fp32(src0_cal, src1_cal, dst, support_bias, bias, K, b, h, m, n);
                }
            }
        }
    }

    //    auto end = std::chrono::high_resolution_clock::now();   // 记录结束时间
    //    std::chrono::duration<double> duration = end - start;  // 计算时间差
    //    std::cout<<duration.count()<<std::endl; // 返回秒数
    return NO_ERROR;
}

ErrorCode mat_mul_fp32_q4_0(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
    /*
    //This is used for test : quantize Q4 here.
    Tensor src1_q4(src1->shape());
    src1_q4.setBackend(src1->backend());
    src1_q4.setDtype(MLLM_TYPE_Q4_0);
    src1_q4.alloc();
    quantize_row_q4_0(src1->hostPtr<float>(), src1_q4.hostPtr<block_q4_0>(), src1->count());
    src1 = &src1_q4;
     */
    assert(src1->dtype() == MLLM_TYPE_Q4_0);

    assert (src0->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0->shape());
    src0_q8.setBackend(src0->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_0);
    src0_q8.alloc();
    quantize_row_q8_0(src0->hostPtr<float>(), src0_q8.hostPtr<block_q8_0>(), src0->count());
    src0 = &src0_q8;
    assert(src0->dtype() == MLLM_TYPE_Q8_0);
    int M = transpose0 ? src0->dimension() : src0->sequence();
    int K = transpose0 ? src0->sequence() : src0->dimension();
    int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = (transpose1 && !transpose0) ? src0 : (transpose0 && !transpose1) ? tensor_trans(src0) : src0;
    Tensor *src1_cal = (transpose1 && !transpose0) ? src1 : (!transpose0 && !transpose1) ? tensor_trans(src1) : src1;
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
        #pragma omp parallel for num_threads(8)
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    vec_dot_q4_0_q8_0(src0_cal->hostPtr<block_q8_0>() + src0_cal->offset(b, h, m, 0)/QK8_0,
                                      src1_cal->hostPtr<block_q4_0>() + src1_cal->offset(b, h, n, 0)/(QK4_0),
                                      dst, support_bias, bias, K, b, h, m, n);
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode mat_mul_fp32_q4_K(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {

    //This is used for test : quantize Q4 here.
    /*
    Tensor src1_q4(src1->shape());
    src1_q4.setBackend(src1->backend());
    src1_q4.setDtype(MLLM_TYPE_Q4_K);
    src1_q4.alloc();
    quantize_row_q4_K(src1->hostPtr<float>(), src1_q4.hostPtr<block_q4_K>(), src1->count());
    src1 = &src1_q4;
    */
    assert(src1->dtype() == MLLM_TYPE_Q4_K);

    assert (src0->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0->shape());
    src0_q8.setBackend(src0->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
    quantize_row_q8_K(src0->hostPtr<float>(), src0_q8.hostPtr<block_q8_K>(), src0->count());
    src0 = &src0_q8;
    assert(src0->dtype() == MLLM_TYPE_Q8_K);
    int M = transpose0 ? src0->dimension() : src0->sequence();
    int K = transpose0 ? src0->sequence() : src0->dimension();
    int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = (transpose1 && !transpose0) ? src0 : (transpose0 && !transpose1) ? tensor_trans(src0) : src0;
    Tensor *src1_cal = (transpose1 && !transpose0) ? src1 : (!transpose0 && !transpose1) ? tensor_trans(src1) : src1;
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            #pragma omp parallel for num_threads(8)
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    vec_dot_q4_K_q8_K(src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0)/QK_K,
                                      src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b,h,n,0)/(QK_K),
                                      dst, support_bias, bias, K, b, h, m, n);
                }
            }
        }
    }
    return NO_ERROR;
}
