//
// Created by ey on 23-10-24.
//

#include "Matmul.hpp"
#include <omp.h>
#include <chrono>

//  COPY FROM GGML
#define MLLM_AVX2_
#define __FMA__
#define GGML_F32_STEP 32
#define GGML_F32_EPR  8
#define GGML_F32_ARR (GGML_F32_STEP/GGML_F32_EPR)
#define GGML_F32x8         __m256
#define GGML_F32x8_ZERO    _mm256_setzero_ps()
#define GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#define GGML_F32x8_LOAD    _mm256_loadu_ps
#define GGML_F32x8_STORE   _mm256_storeu_ps
#if defined(__FMA__)
#define GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
#define GGML_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define GGML_F32x8_ADD     _mm256_add_ps
#define GGML_F32x8_MUL     _mm256_mul_ps
#define GGML_F32x8_REDUCE(res, x)                                 \
{                                                                 \
    int offset = GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));                     \
}
#define GGML_F32x8         __m256
#define GGML_F32_VEC        GGML_F32x8
#define GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x8_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x8_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x8_STORE
#define GGML_F32_VEC_FMA    GGML_F32x8_FMA
#define GGML_F32_VEC_ADD    GGML_F32x8_ADD
#define GGML_F32_VEC_MUL    GGML_F32x8_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE
static void vec_dot_fp32_AVX2__(const int n, float * s, const float * x, const float * y) {
    float sumf = 0.0f;
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

            sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }

    *s = sumf;
}

void vec_dot_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf) {
    float value = 0;
#ifdef MLLM_AVX2_
        vec_dot_fp32_AVX2__(hid_len, &value, src0->ptrAt<float>(batch, head, src0_inf, 0), src1->ptrAt<float>(batch, head, sec1_outf, 0));
#else
        for (int k = 0; k < hid_len; k++) {
            value += src0->dataAt<float>({batch, head, src0_inf, k}) * src1->dataAt<float>({batch, head, sec1_outf, k});
        }
        std::cout<<value1<< ", "<<value<<std::endl;
#endif
    if (support_bias) {
        value += bias->dataAt<float>({0, head, 0, sec1_outf});
    }
    dst->setDataAt<float>({batch, head, src0_inf, sec1_outf}, value);
}

Tensor *tensor_trans(Tensor *src){
    Tensor *dst = new Tensor();
    dst->setBackend(src->backend());
    dst->reshape({src->batch(), src->head(), src->dimension(), src->sequence()});
    dst->setDtype(src->dtype());
    dst->alloc();
    for (int b = 0; b < src->batch(); b++) {
        for (int h = 0; h < src->head(); h++) {
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
ErrorCode mat_mul_fp32_q4_0(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
    Tensor src1_dequantize(src1->shape());
    src1_dequantize.setDtype(src0->dtype());
    src1_dequantize.alloc();
    dequantize_row_q4_0(src1->hostPtr<block_q4_0>(), src1_dequantize.hostPtr<float>(), src1_dequantize.count());
    mat_mul_fp32(src0, &src1_dequantize, dst, support_bias, bias, transpose0, transpose1);
    return NO_ERROR;
}
