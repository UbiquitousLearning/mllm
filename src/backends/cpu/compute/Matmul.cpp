//
// Created by ey on 23-10-24.
//

#include "Matmul.hpp"
#include <pthread.h>

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
    /*
//    src->reshape({src->batch(), src->head(), src->dimension(), src->sequence()});
//    return src;
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
                #pragma omp parallel for num_threads(4)
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
                #pragma omp parallel for num_threads(4)
                for (int m = 0; m < src->dimension(); m++) {
                    dst->setDataAt<float>({b, h, m, n}, src->dataAt<float>({b, h, n, m}));
                }
            }
        }
    }
    return dst;
*/

}


ErrorCode mat_mul_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
    // INPUT: M.K
    // W:K,N
    // OUTPUT:M.N

    //    auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间

    // int M = src0->sequence();
    // int K = src0->dimension();
    // int N = src1->sequence();
    int M = transpose0 ? src0->dimension() : src0->sequence();
    int K = transpose0 ? src0->sequence() : src0->dimension();
    int N = transpose1 ? src1->sequence() : src1->dimension();
    // Tensor *src0_cal = (transpose1 && !transpose0) ? src0 : (transpose0 && !transpose1) ? tensor_trans(src0) : src0;
    // Tensor *src1_cal = (transpose1 && !transpose0) ? src1 : (!transpose0 && !transpose1) ? tensor_trans(src1) : src1;
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    if(!transpose0 && transpose1) {
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                for (int m = 0; m < M; m++) {
                    /*
#pragma omp parallel for num_threads(4)
                    for (int n = 0; n < N; n++) {
                        vec_dot_fp32(src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0),
                                          src1_cal->hostPtr<float>() + src1_cal->offset(b, h, n, 0),
                                          dst, support_bias, bias, K, b, h, m, n);
                    }
                    */
                    int num_blocks = N / blck_0;
                    int remainder = N % blck_0;
                    // 处理完整的块
#pragma omp parallel for num_threads(4)
                    for (int block = 0; block < num_blocks; block++) {
                        for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                            vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                              src1_cal->hostPtr<float>() + src1_cal->offset(b, h, n, 0),
                                              src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0) );
                            if(support_bias){
                                *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                            }
                        }
                    }
                    // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                    for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                        vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<float>() + src1_cal->offset(b, h, n, 0),
                                          src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0) );
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                        }
                    }
                }
            }
        }
    }else if (!transpose0 && !transpose1) {
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                for (int m = 0; m < M; m++) {
                    /*
#pragma omp parallel for num_threads(4)
                    for (int n = 0; n < N; n++) {
                        vec_dot_fp32(src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0),
                                          src1_cal->hostPtr<float>() + src1_cal->offset(b, h, 0, n),
                                          dst, support_bias, bias, K, b, h, m, n);
                    }
                    */
                    int num_blocks = N / blck_0;
                    int remainder = N % blck_0;
                    // 处理完整的块
#pragma omp parallel for num_threads(4)
                    for (int block = 0; block < num_blocks; block++) {
                        for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                            vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                              src1_cal->hostPtr<float>() + src1_cal->offset(b, h, 0, n),
                                              src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0) );
                            if(support_bias){
                                *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                            }
                        }
                    }
                    // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                    for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                        vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<float>() + src1_cal->offset(b, h, 0, n),
                                          src0_cal->hostPtr<float>() + src0_cal->offset(b, h, m, 0) );
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                        }
                    }
                }
            }
        }
    } else {
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                    for (int m = 0; m < M; m++) {
                        /*
#pragma omp parallel for num_threads(4)
                for (int n = 0; n < N; n++) {
                        vec_dot_fp32(src0_cal->hostPtr<float>() + src0_cal->offset(b, h, 0, m),
                                          src1_cal->hostPtr<float>() + src1_cal->offset(b, h, 0, n),
                                          dst, support_bias, bias, K, b, h, m, n);
                    }
                    */
                    int num_blocks = N / blck_0;
                    int remainder = N % blck_0;
                    // 处理完整的块
#pragma omp parallel for num_threads(4)
                    for (int block = 0; block < num_blocks; block++) {
                        for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                            vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                              src1_cal->hostPtr<float>() + src1_cal->offset(b, h, 0, n),
                                              src0_cal->hostPtr<float>() + src0_cal->offset(b, h, 0, m) );
                            if(support_bias){
                                *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                            }
                        }
                    }
                    // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                    for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                        vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<float>() + src1_cal->offset(b, h, 0, n),
                                          src0_cal->hostPtr<float>() + src0_cal->offset(b, h, 0, m) );
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                        }
                    }
                }
            }
        }
    }


    //    auto end = std::chrono::high_resolution_clock::now();   // 记录结束时间
    //    std::chrono::duration<double> duration = end - start;  // 计算时间差
    //    std::cout<<duration.count()<<std::endl; // 返回秒数
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
                for (int m = 0; m < M; m++) {
                    /*
        #pragma omp parallel for num_threads(4)
            for (int n = 0; n < N; n++) {
                    vec_dot_q4_0_q8_0(src0_cal->hostPtr<block_q8_0>() + src0_cal->offset(b, h, m, 0)/QK8_0,
                                      src1_cal->hostPtr<block_q4_0>() + src1_cal->offset(b, h, n, 0)/(QK4_0),
                                      dst, support_bias, bias, K, b, h, m, n);
                }
                    */
                    int num_blocks = N / blck_0;
                    int remainder = N % blck_0;
                    // 处理完整的块
#pragma omp parallel for num_threads(4)
                    for (int block = 0; block < num_blocks; block++) {
                        for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                            vec_dot_q4_0_q8_0(K, dst->ptrAt<float>(b, h, m, n),
                                              src1_cal->hostPtr<block_q4_0>() + src1_cal->offset(b, h, n, 0) / QK4_0,
                                              src0_cal->hostPtr<block_q8_0>() + src0_cal->offset(b, h, m, 0) / QK8_0);
                            if(support_bias){
                                *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                            }
                        }
                    }
                    // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                    for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                        vec_dot_q4_0_q8_0(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<block_q4_0>() + src1_cal->offset(b, h, n, 0) / QK4_0,
                                          src0_cal->hostPtr<block_q8_0>() + src0_cal->offset(b, h, m, 0) / QK8_0);
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
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
    /*
    const int src0_batch = src0_->shape()[0];
    const int src0_head = src0_->shape()[2];
    const int src0_sequence = src0_->shape()[1];
    const int src0_dimension = src0_->shape()[3];
    const int src1_batch = src1->shape()[0];
    const int src1_head = src1->shape()[2];
    const int src1_sequence = src1->shape()[1];
    const int src1_dimension = src1->shape()[3];
    const int dst_batch = dst->shape()[0];
    const int dst_head = dst->shape()[2];
    const int dst_sequence = dst->shape()[1];
    const int dst_dimension = dst->shape()[3];
    */
    assert (src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
    quantize_row_q8_K(src0_->hostPtr<float>(), src0_q8.hostPtr<block_q8_K>(), src0_->count());
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
    auto *src0 = &src0_q8;
    assert(src0->dtype() == MLLM_TYPE_Q8_K);
    int M = src0->sequence();
    int K = src0->dimension();
    int N = src1->sequence();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    /*
    auto src1_ptr = src1_cal->hostPtr<block_q4_K>();
    auto src0_ptr = src0_cal->hostPtr<block_q8_K>();
    auto dst_ptr = dst->hostPtr<float>();
    */
    const int64_t blck_0 = 16;

    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                /*
                #pragma omp parallel for num_threads(4)
                for (int n = 0; n < N; n++) {
                    // vec_dot_q4_K_q8_K(src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K,
                    //                   src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b, h, n, 0) / QK_K,
                    //                   dst, support_bias, bias, K, b, h, m, n);
                    vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                    src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b, h, n, 0) / QK_K,
                                    src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                }
                */

                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
                // 处理完整的块
                #pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                        vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b, h, n, 0) / QK_K,
                                          src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                        }
                    }
                }
                // 处理剩余的元素
                #pragma omp parallel for num_threads(4)
                for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                    vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                      src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b, h, n, 0) / QK_K,
                                      src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                    if(support_bias){
                        *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                    }
                }

                /*
                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
                // 处理完整的块
                #pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                        vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_ptr + (((b * src1_sequence + n) * src1_head + h) * src1_dimension + 0 )/ QK_K,
                                          src0_ptr + (((b * src0_sequence + m) * src0_head + h) * src0_dimension + 0 )/ QK_K);
                    }
                }
                // 处理剩余的元素
                #pragma omp parallel for num_threads(4)
                for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                    vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_ptr + (((b * src1_sequence + n) * src1_head + h) * src1_dimension + 0 )/ QK_K,
                                          src0_ptr + (((b * src0_sequence + m) * src0_head + h) * src0_dimension + 0 )/ QK_K);

                }
                */
            }
        }
    }
    // std::cout<<"quantize_row_q8_K: "<<(end_t1 - start_t1)/1000.0F<<" ms \t";
    // std::cout<<"j: "<<(start_t2 - end_t1)/1000.0F<<" ms \t";
    // std::cout<<"vec_dot_q4_K_q8_K: "<<(end_t2 - start_t2)/1000.0F<<" ms"<<std::endl;
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
            for (int m = 0; m < M; m++) {
                /*
#pragma omp parallel for num_threads(4)
                for (int n = 0; n < N; n++) {
                    vec_dot_q6_K_q8_K(src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K,
                                      src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b, h, n, 0) / QK_K,
                                      dst, support_bias, bias, K, b, h, m, n);
                }
                */
                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
                // 处理完整的块
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0; n++) {
                        vec_dot_q6_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b, h, n, 0) / QK_K,
                                          src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                        if(support_bias){
                            *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                        }
                    }
                }
                // 处理剩余的元素
#pragma omp parallel for num_threads(4)
                for (int n = num_blocks * blck_0; n < num_blocks * blck_0 + remainder; n++) {
                    vec_dot_q6_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                      src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b, h, n, 0) / QK_K,
                                      src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                    if(support_bias){
                        *dst->ptrAt<float>(b,h,m,n) +=  bias->dataAt<float>(0, h, 0, n);
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

