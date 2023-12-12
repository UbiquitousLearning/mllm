//
// Created by ey on 23-10-24.
//

#include "Matmul.hpp"
#include <pthread.h>

/*
#define F32_BLOCK 16
inline void transpose_scalar_block(const float *A, float *B, const int lda, const int ldb, const int block_size = F32_BLOCK) {
    int i;
    int j = 0;
// Cache Aware Transpose
#pragma omp parallel for num_threads(4)
    for (i = 0; i < block_size; i++) {
        for (j = 0; j < block_size; j++) {
            B[j * ldb + i] = A[i * lda + j];
        }
    }
}
Tensor *tensor_trans(Tensor *src) {
//    uint64_t t_start = mllm_time_us();
    Tensor *dst = new Tensor();
    dst->setBackend(src->backend());
    dst->reshape(src->batch(), src->head(), src->dimension(), src->sequence());
    dst->setDtype(src->dtype());
    dst->alloc();
    for (int b = 0; b < src->batch(); b++) {
        for (int h = 0; h < src->head(); h++) {
            for (int n = 0; n < src->sequence(); n++) {
#pragma omp parallel for num_threads(4)
                for (int m = 0; m < src->dimension(); m++) {
                    dst->setDataAt<float>({b, h, m, n}, src->dataAt<float>(b, h, n, m));
                }
            }
        }
    }
//    uint64_t t_end = mllm_time_us();
//    std::cout<<"\n ====  "<<src->name()<<":["<<src->batch()<<","<<src->head()<<","<<src->sequence()<<","<<src->dimension()<<"]"
//        <<" ====  "<< (t_end - t_start)/1000.0F << " ms" << std::endl;
    return dst;
}
*/

ErrorCode mat_mul_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
    int M = transpose0 ? src0->dimension() : src0->sequence();
    int K = transpose0 ? src0->sequence() : src0->dimension();
    int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    if(!transpose0 && transpose1) {
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                int b_1 = b;
                int h_1 = h;
                if(src1->batch()==1 && src1->head()==1){
                    b_1 = 0;
                    h_1 = 0;
                }
                for (int m = 0; m < M; m++) {
                    int num_blocks = N / blck_0;
                    int remainder = N % blck_0;
                    // 处理完整的块
#pragma omp parallel for num_threads(4)
                    for (int block = 0; block < num_blocks; block++) {
                        for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                            vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                         src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, n, 0),
                                         src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0) );
                            if(support_bias){
                                *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                            }
                        }
                    }
                    // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                    for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                        vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                     src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, n, 0),
                                     src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0) );
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    }else if (!transpose0 && !transpose1) {
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                int b_1 = b;
                int h_1 = h;
                if(src1->batch()==1 && src1->head()==1){
                    b_1 = 0;
                    h_1 = 0;
                }
                for (int m = 0; m < M; m++) {
                    int num_blocks = N / blck_0;
                    int remainder = N % blck_0;
                    // 处理完整的块
#pragma omp parallel for num_threads(4)
                    for (int block = 0; block < num_blocks; block++) {
                        for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                            vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                         src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, 0, n),
                                         src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0) );
                            if(support_bias){
                                *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                            }
                        }
                    }
                    // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                    for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                        vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                     src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, 0, n),
                                     src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0) );
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    } else {
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                int b_1 = b;
                int h_1 = h;
                if(src1->batch()==1 && src1->head()==1){
                    b_1 = 0;
                    h_1 = 0;
                }
                for (int m = 0; m < M; m++) {
                    int num_blocks = N / blck_0;
                    int remainder = N % blck_0;
                    // 处理完整的块
#pragma omp parallel for num_threads(4)
                    for (int block = 0; block < num_blocks; block++) {
                        for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                            vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                         src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, 0, n),
                                         src0_cal->hostPtr<float>() + src0_cal->offset(b, h, 0, m) );
                            if(support_bias){
                                *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                            }
                        }
                    }
                    // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                    for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                        vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                     src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, 0, n),
                                     src0_cal->hostPtr<float>() + src0_cal->offset(b, h, 0, m) );
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    }

    return NO_ERROR;
}

ErrorCode mat_mul_fp32_q4_0(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
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

    assert (src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_0);
    src0_q8.alloc();
    //quantize_row_q8_0(src0_->hostPtr<float>(), src0_q8.hostPtr<block_q8_0>(), src0_->count());
    if (src0_->dimension() % QK8_0 == 0) {
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
#pragma omp parallel for num_threads(4)
                for (int s = 0; s < src0_->sequence(); s++) {
                    quantize_row_q8_0(src0_->hostPtr<float>() + src0_->offset(b, h, s, 0),
                                      src0_q8.hostPtr<block_q8_0>() + src0_q8.offset(b, h, s, 0) / QK8_0,
                                      src0_->dimension());
                }
            }
        }
    } else {
        std::cout << "[ERROR]: " << src0_->dimension() << "%" << QK8_0 << "!=0" << std::endl;
        assert(src0_->dimension() % QK8_0 == 0);
    }
    auto *src0 = &src0_q8;
    assert(src0->dtype() == MLLM_TYPE_Q8_0);
    int M = src0->sequence();
    int K = src0->dimension();
    int N = src1->sequence();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            int b_1 = b;
            int h_1 = h;
            if(src1->batch()==1 && src1->head()==1){
                b_1 = 0;
                h_1 = 0;
            }
            for (int m = 0; m < M; m++) {
                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
                // 处理完整的块
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                        vec_dot_q4_0_q8_0(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<block_q4_0>() + src1_cal->offset(b_1, h_1, n, 0) / QK4_0,
                                          src0_cal->hostPtr<block_q8_0>() + src0_cal->offset(b, h, m, 0) / QK8_0);
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
                // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                    vec_dot_q4_0_q8_0(K, dst->ptrAt<float>(b, h, m, n),
                                      src1_cal->hostPtr<block_q4_0>() + src1_cal->offset(b_1, h_1, n, 0) / QK4_0,
                                      src0_cal->hostPtr<block_q8_0>() + src0_cal->offset(b, h, m, 0) / QK8_0);
                    if(support_bias){
                        *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode mat_mul_fp32_q4_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {

    //    uint64_t t_start = mllm_time_us();
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
    assert (src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
//    quantize_row_q8_K(src0_->hostPtr<float>(), src0_q8.hostPtr<block_q8_K>(), src0_->count());
    if (src0_->dimension() % QK_K == 0) {
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
#pragma omp parallel for num_threads(4)
                for (int s = 0; s < src0_->sequence(); s++) {
                    quantize_row_q8_K(src0_->hostPtr<float>() + src0_->offset(b, h, s, 0),
                                      src0_q8.hostPtr<block_q8_K>() + src0_q8.offset(b, h, s, 0) / QK_K,
                                      src0_->dimension());
                }
            }
        }
    } else {
        std::cout << "[ERROR]: " << src0_->dimension() << "%" << QK_K << "!=0" << std::endl;
        assert(src0_->dimension() % QK_K == 0);
    }
    auto *src0 = &src0_q8;
    assert(src0->dtype() == MLLM_TYPE_Q8_K);
    int M = src0->sequence();
    int K = src0->dimension();
    int N = src1->sequence();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;

    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            int b_1 = b;
            int h_1 = h;
            if(src1->batch()==1 && src1->head()==1){
                b_1 = 0;
                h_1 = 0;
            }
            for (int m = 0; m < M; m++) {

                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
// 处理完整的块
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                        vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b_1, h_1, n, 0) / QK_K,
                                          src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
// 处理剩余的元素
#pragma omp parallel for num_threads(4)
                for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                    vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                      src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b_1, h_1, n, 0) / QK_K,
                                      src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                    if(support_bias){
                        *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode mat_mul_fp32_q6_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {

    //    uint64_t t_start = mllm_time_us();
    //This is used for test : quantize Q4 here.
    /*
    Tensor src1_q4(src1->shape());
    src1_q4.setBackend(src1->backend());
    src1_q4.setDtype(MLLM_TYPE_Q4_K);
    src1_q4.alloc();
    quantize_row_q4_K(src1->hostPtr<float>(), src1_q4.hostPtr<block_q4_K>(), src1->count());
    src1 = &src1_q4;
    */
    assert(src1->dtype() == MLLM_TYPE_Q6_K);

    assert (src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
    //    quantize_row_q8_K(src0_->hostPtr<float>(), src0_q8.hostPtr<block_q8_K>(), src0_->count());
    if(src0_->dimension() % QK_K == 0) {
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
#pragma omp parallel for num_threads(4)
                for (int s = 0; s < src0_->sequence(); s++) {
                    quantize_row_q8_K(src0_->hostPtr<float>() + src0_->offset(b, h, s, 0),
                                      src0_q8.hostPtr<block_q8_K>() + src0_q8.offset(b, h, s, 0) / QK_K,
                                      src0_->dimension());
                }
            }
        }
    }else{
        std::cout<<"[ERROR]: "<<src0_->dimension() << "%"<< QK_K <<"!=0"<<std::endl;
        assert(src0_->dimension() % QK_K == 0);
    }
    auto *src0 = &src0_q8;
    assert(src0->dtype() == MLLM_TYPE_Q8_K);
    int M = src0->sequence();
    int K = src0->dimension();
    int N = src1->sequence();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            int b_1 = b;
            int h_1 = h;
            if(src1->batch()==1 && src1->head()==1){
                b_1 = 0;
                h_1 = 0;
            }
            for (int m = 0; m < M; m++) {
                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
                // 处理完整的块
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                        vec_dot_q6_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b_1, h_1, n, 0) / QK_K,
                                          src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
                // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                    vec_dot_q6_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                      src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b_1, h_1, n, 0) / QK_K,
                                      src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                    if(support_bias){
                        *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, 0, 0, n);
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

