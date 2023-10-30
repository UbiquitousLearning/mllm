//
// Created by ey on 23-10-24.
//

#include "Matmul.hpp"


#define F32_BLOCK 16
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
                    vec_dot_fp32(src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0),
                                      src1_cal->hostPtr<float>() + src1_cal->offset(b, h, n, 0),
                                      dst, support_bias, bias, K, b, h, m, n);
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
