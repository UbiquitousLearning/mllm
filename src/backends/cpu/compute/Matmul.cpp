//
// Created by ey on 23-10-24.
//

#include "Matmul.hpp"
#include <pthread.h>

ErrorCode mat_mul_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            const int b_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : b;
            const int h_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : h;
            for (int m = 0; m < M; m++) {
                const int num_blocks = N / blck_0;
                const int remainder = N % blck_0;
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < num_blocks * blck_0 + remainder; n++) {
                        int s_1, d_1;
                        int s_0, d_0;
                        if (!transpose0 && transpose1) {
                            s_1 = n; d_1 = 0; s_0 = m; d_0 = 0;
                        } else if (!transpose0 && !transpose1) {
                            s_1 = 0; d_1 = n; s_0 = m; d_0 = 0;
                        } else {
                            s_1 = 0; d_1 = n; s_0 = 0; d_0 = m;
                        }
                        if(dst->dtypeAt(n,h,m,n) == MLLM_TYPE_F32) {
                            vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                         src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, s_1, d_1),
                                         src0_cal->hostPtr<float>() + src0_cal->offset(b, h, s_0, d_0));
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        }else if (dst->dtypeAt(n,h,m,n) == MLLM_TYPE_F16) {
                            float tmp = 0;
                            vec_dot_fp32(K, &tmp,
                                         src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, s_1, d_1),
                                         src0_cal->hostPtr<float>() + src0_cal->offset(b, h, s_0, d_0));
                            if (support_bias) {
                                *dst->ptrAt<mllm_fp16_t>(b, h, m, n) = MLLM_FP32_TO_FP16(tmp + bias->dataAt<float>(0, 0, 0, n));
                            } else {
                                *dst->ptrAt<mllm_fp16_t>(b, h, m, n) = MLLM_FP32_TO_FP16(tmp);
                            }
                        }else{std::cout<<"Not support type [Matmul]"<<std::endl;}
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode mat_mul_fp32_fp16(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1) {
    assert(src1->dtype() == MLLM_TYPE_F16);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_qf16(src0_->shape());
    src0_qf16.setBackend(src0_->backend());
    src0_qf16.setDtype(MLLM_TYPE_F16);
    src0_qf16.alloc();
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
#pragma omp parallel for num_threads(4)
                for (int s = 0; s < src0_->sequence(); s++) {
                    mllm_fp32_to_fp16_row(src0_->hostPtr<float>() + src0_->offset(b, h, s, 0),
                                      src0_qf16.hostPtr<mllm_fp16_t>() + src0_qf16.offset(b, h, s, 0),
                                      src0_->dimension());
                }
            }
        }
    auto *src0 = &src0_qf16;
    // for(int b=0; b<src0->dimension(); b++) {
    //     std::cout<<MLLM_COMPUTE_FP16_TO_FP32(*src0->ptrAt<mllm_fp16_t>(0, 0, 0, b))<<" ";
    // }
    // std::cout<<std::endl;
    // for(int b=0; b<src1->dimension(); b++) {
    //     std::cout<<MLLM_COMPUTE_FP16_TO_FP32(*src1->ptrAt<mllm_fp16_t>(0, 0, 0, b))<<" ";
    // }
    // std::cout<<std::endl;
    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            const int b_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : b;
            const int h_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : h;
            for (int m = 0; m < M; m++) {
                const int num_blocks = N / blck_0;
                const int remainder = N % blck_0;
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < num_blocks * blck_0 + remainder; n++) {
                        int s_1, d_1;
                        int s_0, d_0;
                        if (!transpose0 && transpose1) {
                            s_1 = n; d_1 = 0; s_0 = m; d_0 = 0;
                        } else if (!transpose0 && !transpose1) {
                            s_1 = 0; d_1 = n; s_0 = m; d_0 = 0;
                        } else {
                            s_1 = 0; d_1 = n; s_0 = 0; d_0 = m;
                        }
                        vec_dot_fp16(K, dst->ptrAt<float>(b, h, m, n),
                                     src1_cal->hostPtr<mllm_fp16_t>() + src1_cal->offset(b_1, h_1, s_1, d_1),
                                     src0_cal->hostPtr<mllm_fp16_t>() + src0_cal->offset(b, h, s_0, d_0));
                        if (support_bias) {
                            *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode mat_mul_fp32_q4_0(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias) {
    assert(src1->dtype() == MLLM_TYPE_Q4_0);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_0);
    src0_q8.alloc();
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
            const int b_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : b;
            const int h_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : h;
            for (int m = 0; m < M; m++) {
                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < num_blocks * blck_0 + remainder; n++) {
                        vec_dot_q4_0_q8_0(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<block_q4_0>() + src1_cal->offset(b_1, h_1, n, 0) / QK4_0,
                                          src0_cal->hostPtr<block_q8_0>() + src0_cal->offset(b, h, m, 0) / QK8_0);
                        if (support_bias) {
                            *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode mat_mul_fp32_q4_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias) {
    assert(src1->dtype() == MLLM_TYPE_Q4_K);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
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
            const int b_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : b;
            const int h_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : h;
            for (int m = 0; m < M; m++) {
                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < num_blocks * blck_0 + remainder; n++) {
                        if(dst->dtypeAt(n,h,m,n) == MLLM_TYPE_F32) {
                            vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                              src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b_1, h_1, n, 0) / QK_K,
                                              src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        } else if (dst->dtypeAt(n,h,m,n) == MLLM_TYPE_F16) {
                            float tmp = 0;
                            vec_dot_q4_K_q8_K(K, &tmp,
                                              src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b_1, h_1, n, 0) / QK_K,
                                              src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                            if (support_bias) {
                                *dst->ptrAt<mllm_fp16_t>(b, h, m, n) = MLLM_FP32_TO_FP16(tmp + bias->dataAt<float>(0, 0, 0, n));
                            } else {
                                *dst->ptrAt<mllm_fp16_t>(b, h, m, n) = MLLM_FP32_TO_FP16(tmp);
                            }
                        }else{std::cout<<"Not support type [Matmul]"<<std::endl;}
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode mat_mul_fp32_q6_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias) {
    assert(src1->dtype() == MLLM_TYPE_Q6_K);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
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
            const int b_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : b;
            const int h_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : h;
            for (int m = 0; m < M; m++) {
                int num_blocks = N / blck_0;
                int remainder = N % blck_0;
#pragma omp parallel for num_threads(4)
                for (int block = 0; block < num_blocks + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < num_blocks * blck_0 + remainder; n++) {
                        if (dst->dtypeAt(n, h, m, n) == MLLM_TYPE_F32) {
                            vec_dot_q6_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                              src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b_1, h_1, n, 0) / QK_K,
                                              src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        } else if (dst->dtypeAt(n, h, m, n) == MLLM_TYPE_F16) {
                            float tmp = 0;
                            vec_dot_q6_K_q8_K(K, &tmp,
                                              src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b_1, h_1, n, 0) / QK_K,
                                              src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);

                            if (support_bias) {
                                *dst->ptrAt<mllm_fp16_t>(b, h, m, n) = MLLM_FP32_TO_FP16(tmp + bias->dataAt<float>(0, 0, 0, n));
                            } else {
                                *dst->ptrAt<mllm_fp16_t>(b, h, m, n) = MLLM_FP32_TO_FP16(tmp);
                            }
                        } else {
                            std::cout << "Not support tupe [Matmul]" << std::endl;
                        }
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

