//
// Created by Rongjie Yi on 23-10-24.
//

#include "Matmul.hpp"
#include "Types.hpp"
#include "VecDotType.hpp"
#include <pthread.h>
#include "SGEMM.hpp"

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

ErrorCode mat_mul(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, int thread_count) {
    // src1 = W  src0 = x
    // transpose0=false  transpose1=true
    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();

    auto src0_dtype = src0->dtype();
    auto src1_dtype = src1->dtype();
    auto vec_dot_type = type_traits[src1_dtype].vec_dot_type;
    auto vec_dot = type_traits[src1_dtype].vec_dot;
    auto x_to_vec_dot_type = type_traits[vec_dot_type].from_float;
    auto from_float_to_mat = type_traits[vec_dot_type].from_float_to_mat;
    mllm_gemv_func const gemv = type_traits[src1_dtype].gemv;
    mllm_gemm_func const gemm = type_traits[src1_dtype].gemm;
    auto blck_size_interleave = type_traits[src1_dtype].blck_size_interleave;

    auto src1_type_size = type_size(src1_dtype);
    auto src1_blck_size = blck_size(src1_dtype);
    auto src0_type_size = type_size(src0->dtype());
    auto src0_blck_size = blck_size(src0->dtype());
#ifdef LLAMAFILE_SGEMM
    if (check_llamafile_sgemm(N, M, K/blck_size(src0->dtype()),src1->dtype(),src0->dtype(),dst->dtype())&&!support_bias){
        const int ld_src1 = src1->sequence_skip_dim();
        const int ld_src0 = src0->sequence_skip_dim();
        const int ld_dst = dst->sequence_skip_dim();
        int is_0 = (src1->batch() == 1 && src1->head() == 1&&src1->batch()!=src0->batch()) ? 0 : 1;
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++){
            for (int64_t h = 0; h < dst->head(); h++){
                for (int id = 0; id < thread_count; id++){
                    llamafile_sgemm(N, M, K/blck_size(src0->dtype()),
                                    (char *)src1->rawHostPtr() + src1->offset(b*is_0, h*is_0, 0, 0) * src1_type_size / src1_blck_size,
                                    ld_src1 / src1_blck_size,
                                    (char *)src0->rawHostPtr() + src0->offset(b, h, 0, 0) * src0_type_size / src0_blck_size,
                                    ld_src0/ src0_blck_size,
                                    (char *)dst->rawHostPtr() + dst->offset(b, h, 0, 0) * type_size(dst->dtype()) / blck_size(dst->dtype()),
                                    ld_dst/blck_size(dst->dtype()),
                                    id, thread_count,
                                    src1->dtype(),
                                    src0->dtype(),
                                    dst->dtype());
                }
            }
        }
        return MLLM_NO_ERROR;
    }
#endif
    auto not_vec_dot_type = src0_dtype != vec_dot_type;
    std::unique_ptr<Tensor> to; // later this tensor will be freed by ~Tensor
    if(not_vec_dot_type){
        // convert x.dtype to vec_dot_type
        // so that we can use vec_dot to calculate dot product
        ASSERT(src0_dtype == MLLM_TYPE_F32); // x should be fp32
        to = std::make_unique<Tensor>(src0->shape());
        to->setBackend(src0->backend());
        to->setDtype(vec_dot_type);
        to->alloc();
//         void *row_src = src0->rawHostPtr();
//         void *row_dst = to->rawHostPtr();
//         auto row_size_src = row_size(src0_dtype, src0->dimension());
//         auto row_size_dst = row_size(vec_dot_type, to->dimension());
//         auto n_row = src0->batch() * src0->head() * src0->sequence();
//         auto n_ele = src0->dimension();
// #pragma omp parallel for num_threads(thread_count)
//         for(int i = 0;i < n_row;i++){ // copy row by row
//             auto row1 = (char *)row_src + i * row_size_src;
//             auto row2 = (char *)row_dst + i * row_size_dst;
//             x_to_vec_dot_type(reinterpret_cast<const float *>(row1), row2, n_ele);
//         }
        int64_t i_processed = 0;
        if (from_float_to_mat && gemv && dst->masterTensor()==nullptr){
            for (int b = 0; b < src0->batch(); b++) {
                for (int h = 0; h < src0->head(); h++) {
#pragma omp parallel for collapse(1) num_threads(thread_count)
                    for (int64_t s = 0; s < src0->sequence() - src0->sequence() % 4; s += 4) {
                            from_float_to_mat(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                            (char *)to->rawHostPtr() + to->offset(b, h, s, 0) * type_size(to->dtype()) / blck_size(to->dtype()),
                                            4, src0->dimension(), blck_size_interleave);
                    }
                    i_processed = src0->sequence() - src0->sequence() % 4;
                }
            }
        }
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                for (int s = i_processed; s < src0->sequence(); s++) {
                    x_to_vec_dot_type(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                      (char *)to->rawHostPtr() + to->offset(b, h, s, 0) * type_size(to->dtype()) / blck_size(to->dtype()),
                                      src0->dimension());
                }
            }
        }
        src0 = to.get();
        src0_dtype = src0->dtype();
        src0_type_size = type_size(src0->dtype());
        src0_blck_size = blck_size(src0->dtype());
    }
        
#ifdef LLAMAFILE_SGEMM
    if (check_llamafile_sgemm(N, M, K/blck_size(src1->dtype()),src1->dtype(),src0->dtype(),dst->dtype())&&!support_bias){
        const int ld_src1 = src1->sequence_skip_dim();
        const int ld_src0 = src0->sequence_skip_dim();
        const int ld_dst = dst->sequence_skip_dim();
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++){
            for (int64_t h = 0; h < dst->head(); h++){
                for (int id = 0; id < thread_count; id++){
                    llamafile_sgemm(N, M, K/blck_size(src1->dtype()),
                                    (char *)src1->rawHostPtr() + src1->offset(b, h, 0, 0) * src1_type_size / src1_blck_size,
                                    ld_src1 / src1_blck_size,
                                    (char *)src0->rawHostPtr() + src0->offset(b, h, 0, 0) * src0_type_size / src0_blck_size,
                                    ld_src0/ src0_blck_size,
                                    (char *)dst->rawHostPtr() + dst->offset(b, h, 0, 0) * type_size(dst->dtype()) / blck_size(dst->dtype()),
                                    ld_dst/blck_size(dst->dtype()),
                                    id, thread_count,
                                    src1->dtype(),
                                    src0->dtype(),
                                    dst->dtype());
                }
            }
        }
        return MLLM_NO_ERROR;
    }
#endif

    if(gemv&&!support_bias){
        int nth=thread_count;
#pragma omp parallel for collapse(1) num_threads(thread_count)
        for (int ith = 0; ith < nth; ith++){
            int64_t i_processed = 0;
            int64_t seq_start = (ith * N) / nth;
            int64_t seq_end   = ((ith + 1) * N) / nth;
            if (gemm && (M > 3) && dst->masterTensor()==nullptr) {
                gemm(K,  dst->hostPtr<float>() +  dst->offset(0, 0, 0, seq_start),
                    N, (char *)src1->rawHostPtr()+ src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                    (char *)src0->rawHostPtr(), M - M % 4, N/nth);
                i_processed = M - M % 4;
            }
            for (int iter = i_processed; iter < M; iter++) { //M-M%4
                gemv(K, dst->hostPtr<float>() +  dst->offset(0, 0, iter, seq_start), 
                    N, (char *)src1->rawHostPtr()+ src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size, 
                    (char *)src0->rawHostPtr() + src0->offset(0, 0, iter, 0) * src0_type_size / src0_blck_size,
                    1,  N/nth);
            }
        }
        return MLLM_NO_ERROR;
    }
    
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    int is_0 = (src1->batch() == 1 && src1->head() == 1&&src1->batch()!=src0->batch()) ? 0 : 1;
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
                        float tmp = 0;
                        vec_dot(K, &tmp,
                                (char *)src1_cal->rawHostPtr() + src1_cal->offset(b*is_0, h*is_0, s_1, d_1) * src1_type_size / src1_blck_size,
                                (char *)src0_cal->rawHostPtr() + src0_cal->offset(b, h, s_0, d_0) * src0_type_size / src0_blck_size);
                        if(dst->dtypeAt(b,h,m,n) == MLLM_TYPE_F32) {
                            dst->setDataAt<float>(b, h, m, n, tmp);
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        }else if(dst->dtypeAt(b,h,m,n) == MLLM_TYPE_F16) {
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
/*
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
*/


ErrorCode mat_mul_elastic(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, 
                            int activate_input_dim, int activate_output_dim, 
                            bool transpose0, bool transpose1, int thread_count) {
    // src1 = W  src0 = x
    // transpose0=false  transpose1=true
    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();

    auto src0_dtype = src0->dtype();
    auto src1_dtype = src1->dtype();
    auto vec_dot_type = type_traits[src1_dtype].vec_dot_type;
    auto vec_dot = type_traits[src1_dtype].vec_dot;
    auto x_to_vec_dot_type = type_traits[vec_dot_type].from_float;

    auto from_float_to_mat = type_traits[vec_dot_type].from_float_to_mat;
    mllm_gemv_func const gemv = type_traits[src1_dtype].gemv;
    mllm_gemm_func const gemm = type_traits[src1_dtype].gemm;
    auto blck_size_interleave = type_traits[src1_dtype].blck_size_interleave;
    auto src1_type_size = type_size(src1_dtype);
    auto src1_blck_size = blck_size(src1_dtype);
    auto src0_type_size = type_size(src0->dtype());
    auto src0_blck_size = blck_size(src0->dtype());

    int use_N = (activate_output_dim == -1) ? N : activate_output_dim;
    int use_K = (activate_input_dim == -1) ? K : activate_input_dim;

    if (check_llamafile_sgemm(use_N, M, use_K/blck_size(src0->dtype()),src1->dtype(),src0->dtype(),dst->dtype())){
        const int ld_src1 = src1->sequence_skip_dim();
        const int ld_src0 = src0->sequence_skip_dim();
        const int ld_dst = dst->sequence_skip_dim();
        int is_0 = (src1->batch() == 1 && src1->head() == 1) ? 0 : 1;
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++){
            for (int64_t h = 0; h < dst->head(); h++){
                for (int id = 0; id < thread_count; id++){
                    llamafile_sgemm(use_N, M, use_K/blck_size(src0->dtype()),
                                    (char *)src1->rawHostPtr() + src1->offset(b*is_0, h*is_0, 0, 0) * src1_type_size / src1_blck_size,
                                    ld_src1,
                                    (char *)src0->rawHostPtr() + src0->offset(b, h, 0, 0) * src0_type_size / src0_blck_size,
                                    ld_src0,
                                    (char *)dst->rawHostPtr() + dst->offset(b, h, 0, 0) * type_size(dst->dtype()) / blck_size(dst->dtype()),
                                    ld_dst,
                                    id, thread_count,
                                    src1->dtype(),
                                    src0->dtype(),
                                    dst->dtype());
                }
            }
        }
        return MLLM_NO_ERROR;
    }

    auto not_vec_dot_type = src0_dtype != vec_dot_type;
    std::unique_ptr<Tensor> to; // later this tensor will be freed by ~Tensor
    if(not_vec_dot_type){
        // convert x.dtype to vec_dot_type
        // so that we can use vec_dot to calculate dot product
        ASSERT(src0_dtype == MLLM_TYPE_F32); // x should be fp32
        to = std::make_unique<Tensor>(src0->shape());
        to->setBackend(src0->backend());
        to->setDtype(vec_dot_type);
        to->alloc();
        int64_t i_processed = 0;
        if (from_float_to_mat && gemv && dst->masterTensor()==nullptr){
            for (int b = 0; b < src0->batch(); b++) {
                for (int h = 0; h < src0->head(); h++) {
#pragma omp parallel for collapse(1) num_threads(thread_count)
                    for (int64_t s = 0; s < src0->sequence() - src0->sequence() % 4; s += 4) {
                            from_float_to_mat(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                            (char *)to->rawHostPtr() + to->offset(b, h, s, 0) * type_size(to->dtype()) / blck_size(to->dtype()),
                                            4, src0->dimension(), blck_size_interleave);
                    }
                    i_processed = src0->sequence() - src0->sequence() % 4;
                }
            }
        }
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < src0->batch(); b++) {
            for (int h = 0; h < src0->head(); h++) {
                for (int s = i_processed; s < src0->sequence(); s++) {
                    x_to_vec_dot_type(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                      (char *)to->rawHostPtr() + to->offset(b, h, s, 0) * type_size(to->dtype()) / blck_size(to->dtype()),
                                      src0->dimension());
                }
            }
        }
        src0 = to.get();
        src0_dtype = src0->dtype();
        src0_type_size = type_size(src0->dtype());
        src0_blck_size = blck_size(src0->dtype());
    }
        
#ifdef LLAMAFILE_SGEMM
    if (check_llamafile_sgemm(N, M, use_K/blck_size(src1->dtype()),src1->dtype(),src0->dtype(),dst->dtype())&&!support_bias){
        const int ld_src1 = src1->sequence_skip_dim();
        const int ld_src0 = src0->sequence_skip_dim();
        const int ld_dst = dst->sequence_skip_dim();
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++){
            for (int64_t h = 0; h < dst->head(); h++){
                for (int id = 0; id < thread_count; id++){
                    llamafile_sgemm(N, M, use_K/blck_size(src1->dtype()),
                                    (char *)src1->rawHostPtr() + src1->offset(b, h, 0, 0) * src1_type_size / src1_blck_size,
                                    ld_src1 / src1_blck_size,
                                    (char *)src0->rawHostPtr() + src0->offset(b, h, 0, 0) * src0_type_size / src0_blck_size,
                                    ld_src0/ src0_blck_size,
                                    (char *)dst->rawHostPtr() + dst->offset(b, h, 0, 0) * type_size(dst->dtype()) / blck_size(dst->dtype()),
                                    ld_dst/blck_size(dst->dtype()),
                                    id, thread_count,
                                    src1->dtype(),
                                    src0->dtype(),
                                    dst->dtype());
                }
            }
        }
        return MLLM_NO_ERROR;
    }
#endif

    if(gemv&&!support_bias){
        int nth=thread_count;
#pragma omp parallel for collapse(1) num_threads(thread_count)
        for (int ith = 0; ith < nth; ith++){
            int64_t i_processed = 0;
            int64_t seq_start = (ith * N) / nth;
            int64_t seq_end   = ((ith + 1) * N) / nth;
            if (gemm && (M > 3) && dst->masterTensor()==nullptr) {
                gemm(use_K,  dst->hostPtr<float>() +  dst->offset(0, 0, 0, seq_start),
                    N, (char *)src1->rawHostPtr()+ src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                    (char *)src0->rawHostPtr(), M - M % 4, N/nth);
                i_processed = M - M % 4;
            }
            for (int iter = i_processed; iter < M; iter++) { //M-M%4
                gemv(use_K, dst->hostPtr<float>() +  dst->offset(0, 0, iter, seq_start), 
                    N, (char *)src1->rawHostPtr()+ src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size, 
                    (char *)src0->rawHostPtr() + src0->offset(0, 0, iter, 0) * src0_type_size / src0_blck_size,
                    1,  N/nth);
            }
        }
        return MLLM_NO_ERROR;
    }

    
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    int is_0 = (src1->batch() == 1 && src1->head() == 1) ? 0 : 1;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int block = 0; block < use_N / blck_0 + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < use_N; n++) {
                        int s_1, d_1;
                        int s_0, d_0;
                        if (!transpose0 && transpose1) {
                            s_1 = n; d_1 = 0; s_0 = m; d_0 = 0;
                        } else if (!transpose0 && !transpose1) {
                            s_1 = 0; d_1 = n; s_0 = m; d_0 = 0;
                        } else {
                            s_1 = 0; d_1 = n; s_0 = 0; d_0 = m;
                        }
                        float tmp = 0;
                        vec_dot(use_K, &tmp,
                                (char *)src1_cal->rawHostPtr() + src1_cal->offset(b*is_0, h*is_0, s_1, d_1) * src1_type_size / src1_blck_size,
                                (char *)src0_cal->rawHostPtr() + src0_cal->offset(b, h, s_0, d_0) * src0_type_size / src0_blck_size);
                        if(dst->dtypeAt(b,h,m,n) == MLLM_TYPE_F32) {
                            dst->setDataAt<float>(b, h, m, n, tmp);
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        }else if(dst->dtypeAt(b,h,m,n) == MLLM_TYPE_F16) {
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