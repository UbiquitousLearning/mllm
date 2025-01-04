//
// Created by Rongjie Yi on 23-10-24.
//

#include "MatmulSparse.hpp"
#include "Types.hpp"
#include "VecDotType.hpp"
// #include <pthread.h>
#include "SGEMM.hpp"
#include <cstdlib>

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <omp.h>
#endif

#define ASSERT(x)                                                                \
    do {                                                                         \
        if (!(x)) {                                                              \
            fflush(stdout);                                                      \
            fprintf(stderr, "MLLM_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort();                                                             \
        }                                                                        \
    } while (0)

ErrorCode sparse_mat_mul_id(Tensor *x, Tensor *W, Tensor *ids, Tensor *dst, int thread_count) {
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
    if (not_vec_dot_type) {
        // convert x.dtype to vec_dot_type
        // so that we can use vec_dot to calculate dot product
        ASSERT(x_dtype == MLLM_TYPE_F32); // x should be fp32
        to = std::make_unique<Tensor>(x->shape());
        to->setBackend(x->backend());
        to->setDtype(vec_dot_type);
        to->alloc();
        to->setName(x->name() + "-vec_dot");
        void *row_src = x->rawHostPtr();
        void *row_dst = to->rawHostPtr();
        auto row_size_src = row_size(x_dtype, x->dimension());
        auto row_size_dst = row_size(vec_dot_type, to->dimension());
        auto n_row = x->batch() * x->head() * x->sequence();
        auto n_ele = x->dimension();
#pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < n_row; i++) { // copy row by row
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
    auto x_row_offset = (x->offset(0, 0, 1, 0) - x->offset(0, 0, 0, 0)) * x_type_size
                        / x_blck_size; // two rows may not be contiguous; layout: <b s h d>
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // fill output M*N matrix
            auto x_row =
                (char *)x->rawHostPtr() + x->offset(b, h, 0, 0) * x_type_size / x_blck_size;
            auto b_W = b % B_W;
            auto h_W = h % H_W;
            for (int m = 0; m < M; m++) {
#pragma omp parallel for num_threads(thread_count)
                for (int x_block = 0; x_block < N; x_block += blck) {
                    for (int n = x_block; n < x_block + blck && n < N; n++) {
                        // predictor says that there is no need to calculate this position
                        if (ids->dataAt<float>(b, h, m, n) <= 0.0) {
                            dst->setDataAt<float>(b, h, m, n, 0.0);
                            continue;
                        }

                        float tmp;

                        vec_dot(K, &tmp,
                                (char *)W->rawHostPtr()
                                    + W->offset(b_W, h_W, n, 0) * w_type_size
                                          / w_blck_size, // cannot calc W_row like x_row, cause
                                                         // b_W,h_W may not be contiguous
                                x_row);
                        dst->setDataAt<float>(
                            b, h, m, n, tmp); // it seems that currently activation can only be fp32
                    }
                }
                x_row = x_row + x_row_offset;
            }
        }
    }
    if (not_vec_dot_type) to->free();
    return MLLM_NO_ERROR;
}

ErrorCode mat_mul_sparse(Tensor *x, Tensor *W, Tensor *dst, int thread_count) {
    /* dst = x * W
     * x: [..., M, K]
     * W: [..., K, N]
     * dst: [..., M, N]
     * we calculate x * W row by row
     * each row can be calc by: Multiply each element in a row of x by the corresponding row in W,
     * and then sum them up. due to the sparsity, we know that most element of x is 0. so we don't
     * need to calc those row
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
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            auto b_W = b % B_W;
            auto h_W = h % H_W;
#pragma omp parallel for num_threads( \
        thread_count) // can not put above for(int n = 0;n < N;n++). that will cause accessing dst
                      // line n at the same time
            for (int m = 0; m < M; m++) {
                auto *fill_row = dst->hostPtr<float>() + dst->offset(b, h, m, 0);
                memset(fill_row, 0, N * sizeof(float));
                for (int k = 0; k < K; k++) {
                    auto alpha = x->dataAt<float>(b, h, m, k);
                    if (alpha != 0.0) {
                        add_row_to(N,
                                   (char *)W->rawHostPtr()
                                       + W->offset(b_W, h_W, k, 0) * w_type_size / w_blck_size,
                                   fill_row, alpha);
                    }
                }
            }
        }
    }

    return MLLM_NO_ERROR;
}