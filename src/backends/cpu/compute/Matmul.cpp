//
// Created by ey on 23-10-24.
//

#include "Matmul.hpp"
#include <chrono>
#ifdef __ARM_NEON
#include "../neon/Neon.hpp"
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
static void vec_dot_fp32_avx2(const int n, float *__restrict s, const float *__restrict x, const float *__restrict y) {
    float sumf = 0.0f;
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
        const __m256 d = _mm256_set1_ps( x[i].d * y[i].d);

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
//TODO THIS IS WRONG CODE
void vec_dot_q4_0_q8_0(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf) {
    float value = 0;
#ifdef __AVX2__
    vec_dot_q4_0_q8_0_avx(hid_len, &value, src1->ptrAt<block_q8_0>(batch, head, src0_inf, 0), src0->ptrAt<block_q4_0>(batch, head, sec1_outf, 0));
#elif defined(__ARM_NEON)
    vec_dot_q4_0_q8_0_arm(hid_len, &value, src1->ptrAt<block_q8_0>(batch, head, src0_inf, 0), src0->ptrAt<block_q4_0>(batch, head, sec1_outf, 0));
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
//TODO CHECK
ErrorCode mat_mul_fp32_q4_0(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
//    Tensor src1_dequantize(src1->shape());
//    src1_dequantize.setDtype(src0->dtype());
//    src1_dequantize.alloc();
//    dequantize_row_q4_0(src1->hostPtr<block_q4_0>(), src1_dequantize.hostPtr<float>(), src1_dequantize.count());
//    mat_mul_fp32(src0, &src1_dequantize, dst, support_bias, bias, transpose0, transpose1);

    Tensor src0_dequantize(src1->shape());
    src0_dequantize.setDtype(MLLM_TYPE_Q8_0);
    src0_dequantize.alloc();
    quantize_row_q8_0(src0->hostPtr<float>(), src0_dequantize.hostPtr<block_q8_0>(), src0_dequantize.count());

    int M = transpose0 ? src0->dimension() : src0->sequence();
    int K = transpose0 ? src0->sequence() : src0->dimension();
    int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = (transpose1 && !transpose0) ? &src0_dequantize : (transpose0 && !transpose1) ? tensor_trans(&src0_dequantize) : &src0_dequantize;
    Tensor *src1_cal = (transpose1 && !transpose0) ? src1 : (!transpose0 && !transpose1) ? tensor_trans(src1) : src1;
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
        #pragma omp parallel for num_threads(8)
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    //quantize_row_q8_0(src0->ptrAt<float>(b, h, m, 0), src0_dequantize.ptrAt<block_q8_0>(b, h, m, 0), K);
                    vec_dot_q4_0_q8_0(src0_cal, src1_cal, dst, support_bias, bias, K, b, h, m, n);
                }
            }
        }
    }
    return NO_ERROR;
}
