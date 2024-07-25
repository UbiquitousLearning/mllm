//
// Created by Rongjie Yi on 23-10-24.
//

#include "Matmul.hpp"
#include "VecDotType.hpp"
#include <pthread.h>

#define ASSERT(x) \
    do { \
        if (!(x)) { \
            fflush(stdout); \
            fprintf(stderr, "MLLM_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

ErrorCode sparse_mat_mul_id(Tensor *x, Tensor *W, Tensor *ids, Tensor *dst, int thread_count){
    /*
     *  dst = x * W^T
     *  x: [..., M, K]
     *  W: [..., N, K]
     *  dst: [..., M, N]
     *  ids: [..., M, N] indicate which column to use in W^T
     *  if ids[..., a,b] <= threshold then dst[..., M, N] should be 0(no need to calculate)
     *
     *  either x.dtype == W.vec_dot_type or x.dtype == MLLM_TYPE_F32
     *  if x.dtype == MLLM_TYPE_F32 and x.dtype != W.vec_dot_type
     *  then we will convert x.dtype to W.vec_dot_type and then calculate
     * */
    const int M = x->sequence();
    const int K = x->dimension();
    const int N = W->sequence();

    ASSERT(W->dimension() == K);
    ASSERT(ids->sequence() == M);
    ASSERT(ids->dimension() == N);
    ASSERT(ids->dtype() == MLLM_TYPE_F32);

    auto B = x->batch();
    auto H = x->head();
    auto B_W = W->batch();
    auto H_W = W->head();
    ASSERT(ids->batch() == B);
    ASSERT(ids->head() == H);

    auto x_dtype = x->dtype();
    auto W_dtype = W->dtype();
    auto vec_dot_type = type_traits[W_dtype].vec_dot_type;
    auto vec_dot = type_traits[W_dtype].vec_dot;
    auto x_to_vec_dot_type = type_traits[vec_dot_type].from_float;
    auto not_vec_dot_type = x_dtype != vec_dot_type;
    std::unique_ptr<Tensor> to; // later this tensor will be freed by ~Tensor
    if(not_vec_dot_type){
        // convert x.dtype to vec_dot_type
        // so that we can use vec_dot to calculate dot product
        ASSERT(x_dtype == MLLM_TYPE_F32); // x should be fp32
        to = std::make_unique<Tensor>(x->shape());
        to->setBackend(x->backend());
        to->setDtype(vec_dot_type);
        to->alloc();
        void *row_src = x->rawHostPtr();
        void *row_dst = to->rawHostPtr();
        auto row_size_src = row_size(x_dtype, x->dimension());
        auto row_size_dst = row_size(vec_dot_type, to->dimension());
        auto n_row = x->batch() * x->head() * x->sequence();
        auto n_ele = x->dimension();
#pragma omp parallel for num_threads(thread_count)
        for(int i = 0;i < n_row;i++){ // copy row by row
            auto row1 = (char *)row_src + i * row_size_src;
            auto row2 = (char *)row_dst + i * row_size_dst;
            x_to_vec_dot_type(reinterpret_cast<const float *>(row1), row2, n_ele);
        }
        x = to.get();
        x_dtype = vec_dot_type;
    }

    const auto x_type_size = type_size(x_dtype);
    const auto x_blck_size = blck_size(x_dtype);
    const auto w_type_size = type_size(W_dtype);
    const auto w_blck_size = blck_size(W_dtype);
    const auto blck = 16;
    auto x_row_offset = (x->offset(0,0,1,0) - x->offset(0,0,0,0)) * x_type_size / x_blck_size; // two rows may not be contiguous; layout: <b s h d>
    for(int b = 0; b < B;b++){
        for(int h = 0;h < H;h++){
            // fill output M*N matrix
            auto x_row = (char *)x->rawHostPtr() + x->offset(b,h,0,0) * x_type_size / x_blck_size;
            auto b_W = b % B_W;
            auto h_W = h % H_W;
            for(int m = 0;m < M;m++){
#pragma omp parallel for num_threads(thread_count)
                for(int x_block=0; x_block < N; x_block += blck) {
                    for (int n = x_block; n < x_block + blck && n < N; n++) {
                        // predictor says that there is no need to calculate this position
                        if (ids->dataAt<float>(b, h, m, n) <= 0.0) {
                            dst->setDataAt<float>(b, h, m, n, 0.0);
                            continue;
                        }

                        float tmp;

                        vec_dot(K,
                                &tmp,
                                (char *)W->rawHostPtr() + W->offset(b_W, h_W, n, 0) * w_type_size / w_blck_size, // cannot calc W_row like x_row, cause b_W,h_W may not be contiguous
                                x_row);
                        dst->setDataAt<float>(b, h, m, n, tmp); // it seems that currently activation can only be fp32
                    }
                }
                x_row = x_row + x_row_offset;
            }

        }
    }
    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_sparse(Tensor *x, Tensor *W, Tensor *dst, int thread_count){
    /* dst = x * W
     * x: [..., M, K]
     * W: [..., K, N]
     * dst: [..., M, N]
     * we calculate x * W row by row
     * each row can be calc by: Multiply each element in a row of x by the corresponding row in W, and then sum them up.
     * due to the sparsity, we know that most element of x is 0.
     * so we don't need to calc those row
     * */
    auto W_dtype = W->dtype();
    ASSERT(x->dtype() == MLLM_TYPE_F32);
    auto M = x->sequence();
    auto K = x->dimension();
    auto N = W->dimension();
    ASSERT(W->sequence() == K);
    ASSERT(dst->batch() == x->batch());
    ASSERT(dst->head() == x->head());
    ASSERT(dst->sequence() == M);
    ASSERT(dst->dimension() == N);

    auto B = x->batch();
    auto H = x->head();
    auto B_W = W->batch();
    auto H_W = W->head();
    auto add_row_to = type_traits[W_dtype].add_row_to;
    const auto w_type_size = type_size(W_dtype);
    const auto w_blck_size = blck_size(W_dtype);
    for(int b = 0; b < B; b ++){
        for(int h = 0; h < H; h ++){
            auto b_W = b % B_W;
            auto h_W = h % H_W;
#pragma omp parallel for num_threads(thread_count) // can not put above for(int n = 0;n < N;n++). that will cause accessing dst line n at the same time
            for(int m = 0; m < M;m++){
                auto fill_row = dst->hostPtr<float>() + dst->offset(b,h,m,0);
                memset(fill_row,
                       0,
                       N * sizeof(float));
                for(int k = 0;k < K;k++){
                    auto alpha = x->dataAt<float>(b,h,m,k);
                    if(alpha != 0.0){
                        add_row_to(N,
                                   (char *)W->rawHostPtr() + W->offset(b_W,h_W,k,0) * w_type_size / w_blck_size,
                                   fill_row,
                                   alpha);
                    }
                }
            }

        }
    }

    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, int thread_count) {
    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    int is_0 = (src1->batch() == 1 && src1->head() == 1) ? 0 : 1;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int block = 0; block < N / blck_0 + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < N; n++) {
                        int s_1, d_1;
                        int s_0, d_0;
                        if (!transpose0 && transpose1) {
                            s_1 = n; d_1 = 0; s_0 = m; d_0 = 0;
                        } else if (!transpose0 && !transpose1) {
                            s_1 = 0; d_1 = n; s_0 = m; d_0 = 0;
                        } else {
                            s_1 = 0; d_1 = n; s_0 = 0; d_0 = m;
                        }
                        if(dst->dtypeAt(b,h,m,n) == MLLM_TYPE_F32) {
                            vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                         src1_cal->hostPtr<float>() + src1_cal->offset(b*is_0, h*is_0, s_1, d_1),
                                         src0_cal->hostPtr<float>() + src0_cal->offset(b, h, s_0, d_0));
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        }else if (dst->dtypeAt(b,h,m,n) == MLLM_TYPE_F16) {
                            float tmp = 0;
                            vec_dot_fp32(K, &tmp,
                                         src1_cal->hostPtr<float>() + src1_cal->offset(b*is_0, h*is_0, s_1, d_1),
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
    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_fp32_fp16(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, int thread_count) {
    assert(src1->dtype() == MLLM_TYPE_F16);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_qf16(src0_->shape());
    src0_qf16.setBackend(src0_->backend());
    src0_qf16.setDtype(MLLM_TYPE_F16);
    src0_qf16.alloc();
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
                for (int s = 0; s < src0_->sequence(); s++) {
                    mllm_fp32_to_fp16_row(src0_->hostPtr<float>() + src0_->offset(b, h, s, 0),
                                      src0_qf16.hostPtr<mllm_fp16_t>() + src0_qf16.offset(b, h, s, 0),
                                      src0_->dimension());
                }
            }
        }
    auto *src0 = &src0_qf16;
    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    int is_0 = (src1->batch() == 1 && src1->head() == 1) ? 0 : 1;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int block = 0; block < N / blck_0 + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < N; n++) {
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
                                     src1_cal->hostPtr<mllm_fp16_t>() + src1_cal->offset(b*is_0, h*is_0, s_1, d_1),
                                     src0_cal->hostPtr<mllm_fp16_t>() + src0_cal->offset(b, h, s_0, d_0));
                        if (support_bias) {
                            *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    }
    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_fp32_q4_0(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int thread_count) {
    assert(src1->dtype() == MLLM_TYPE_Q4_0);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_0);
    src0_q8.alloc();
    if (src0_->dimension() % QK8_0 == 0) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
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
    int is_0 = (src1->batch() == 1 && src1->head() == 1) ? 0 : 1;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int block = 0; block < N / blck_0 + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < N; n++) {
                        vec_dot_q4_0_q8_0(K, dst->ptrAt<float>(b, h, m, n),
                                          src1_cal->hostPtr<block_q4_0>() + src1_cal->offset(b*is_0, h*is_0, n, 0) / QK4_0,
                                          src0_cal->hostPtr<block_q8_0>() + src0_cal->offset(b, h, m, 0) / QK8_0);
                        if (support_bias) {
                            *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    }
    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_fp32_q4_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int thread_count) {
    assert(src1->dtype() == MLLM_TYPE_Q4_K);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
    if (src0_->dimension() % QK_K == 0) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
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
    int is_0 = (src1->batch() == 1 && src1->head() == 1) ? 0 : 1;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int block = 0; block < N / blck_0 + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < N; n++) {
                        if(dst->dtypeAt(b,h,m,n) == MLLM_TYPE_F32) {
                            vec_dot_q4_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                              src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b*is_0, h*is_0, n, 0) / QK_K,
                                              src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        } else if (dst->dtypeAt(b,h,m,n) == MLLM_TYPE_F16) {
                            float tmp = 0;
                            vec_dot_q4_K_q8_K(K, &tmp,
                                              src1_cal->hostPtr<block_q4_K>() + src1_cal->offset(b*is_0, h*is_0, n, 0) / QK_K,
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
    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_fp32_q6_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int thread_count) {
    assert(src1->dtype() == MLLM_TYPE_Q6_K);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_q8(src0_->shape());
    src0_q8.setBackend(src0_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
    if (src0_->dimension() % QK_K == 0) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
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
    int is_0 = (src1->batch() == 1 && src1->head() == 1) ? 0 : 1;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int block = 0; block <  N / blck_0 + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n <  N; n++) {
                        if (dst->dtypeAt(n, h, m, n) == MLLM_TYPE_F32) {
                            vec_dot_q6_K_q8_K(K, dst->ptrAt<float>(b, h, m, n),
                                              src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b*is_0, h*is_0, n, 0) / QK_K,
                                              src0_cal->hostPtr<block_q8_K>() + src0_cal->offset(b, h, m, 0) / QK_K);
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        } else if (dst->dtypeAt(n, h, m, n) == MLLM_TYPE_F16) {
                            float tmp = 0;
                            vec_dot_q6_K_q8_K(K, &tmp,
                                              src1_cal->hostPtr<block_q6_K>() + src1_cal->offset(b*is_0, h*is_0, n, 0) / QK_K,
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
    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_i8(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, int thread_count, float scale1, float scale2) {
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
#pragma omp parallel for num_threads(thread_count)
                for (int block = 0; block < num_blocks + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < num_blocks * blck_0 + remainder; n++) {
                        int s_1, d_1;
                        int s_0, d_0;
                        if (!transpose0 && transpose1) {
                            s_1 = n;
                            d_1 = 0;
                            s_0 = m;
                            d_0 = 0;
                        } else if (!transpose0 && !transpose1) {
                            s_1 = 0;
                            d_1 = n;
                            s_0 = m;
                            d_0 = 0;
                        } else {
                            s_1 = 0;
                            d_1 = n;
                            s_0 = 0;
                            d_0 = m;
                        }

                        vec_dot_q8_0_q8_0(K, dst->ptrAt<float>(b, h, m, n), src1_cal->hostPtr<int8_t>() + src1_cal->offset(b_1, h_1, s_1, d_1), src0_cal->hostPtr<int8_t>() + src0_cal->offset(b, h, s_0, d_0), scale1, scale2);
                        if (support_bias) {
                            *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    }
    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_fp32_i8(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, int thread_count, float scale2) {
    assert(src0->dtype() == MLLM_TYPE_F32);
    assert(src1->dtype() == MLLM_TYPE_I8);

    Tensor src1_fp(src1->batch(), src1->head(), src1->dimension(), src1->sequence());
    src1_fp.setBackend(src1->backend());
    src1_fp.setDtype(MLLM_TYPE_F32);
    src1_fp.alloc();
    // if (src0->dimension() % QK8_0 == 0) {
        
    // } else {
    //     std::cout << "[ERROR]: " << src0->dimension() << "%" << QK8_0 << "!=0" << std::endl;
    //     src0->printShape();
    //     assert(src0->dimension() % QK8_0 == 0);
    // }

    float *src1_fpp = src1_fp.hostPtr<float>();

    for (int b = 0; b < src1->batch(); b++) {
        for (int d = 0; d < src1->dimension(); d++) {
            // #pragma omp parallel for num_threads(thread_count)
            for (int h = 0; h < src1->head(); h++) {
                for (int s = 0; s < src1->sequence(); s++) {
                    dequantize_row_i8(src1->hostPtr<int8_t>() + (b * src1->dimension() * src1->head() * src1->sequence() + s * src1->head() * src1->dimension()+ h * src1->dimension() + d),  src1_fpp, 1, scale2);

                    src1_fpp ++;
                }
            }
        }
    }

    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = &src1_fp;
    const int64_t blck_0 = 16;

    assert(dst->dtype() == MLLM_TYPE_F32);
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            const int b_1 = (src1_fp.batch() == 1 && src1_fp.head() == 1) ? 0 : b;
            const int h_1 = (src1_fp.batch() == 1 && src1_fp.head() == 1) ? 0 : h;
            for (int m = 0; m < M; m++) {
                const int num_blocks = N / blck_0;
                const int remainder = N % blck_0;
#pragma omp parallel for num_threads(thread_count)
                for (int block = 0; block < num_blocks + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < num_blocks * blck_0 + remainder; n++) {
                        int s_1, d_1;
                        int s_0, d_0;
                        if (!transpose0 && !transpose1) {
                            s_1 = n;
                            d_1 = 0;
                            s_0 = m;
                            d_0 = 0;
                        } else {
                            std::cout << "NO support" << std::endl;
                        }
                        vec_dot_fp32(K, dst->ptrAt<float>(b, h, m, n),
                                     src1_cal->hostPtr<float>() + src1_cal->offset(b_1, h_1, s_1, d_1),
                                     src0_cal->hostPtr<float>() + src0_cal->offset(b, h, s_0, d_0));
                        if (support_bias) {
                            *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                        }
                    }
                }
            }
        }
    }
    return MLLM_NO_ERROR;
}