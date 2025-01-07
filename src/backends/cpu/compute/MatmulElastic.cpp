//
// Created by Rongjie Yi on 23-10-24.
//

#include "MatmulElastic.hpp"
#include "Types.hpp"
#include "VecDotType.hpp"
// #include <pthread.h>
#include "SGEMM.hpp"
#include <cassert>
#include <cstdlib>

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <omp.h>
#endif

ErrorCode mat_mul_elastic(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias,
                          int activate_input_dim, int activate_output_dim, bool transpose0,
                          bool transpose1, int thread_count) {
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

#ifdef LLAMAFILE_SGEMM
    int ld_src1 = src1->sequenceSkipDim();
    int ld_src0 = src0->sequenceSkipDim();
    int ld_dst = dst->sequenceSkipDim();
    if (check_llamafile_sgemm(N, M, K / blck_size(src0->dtype()), src1->dtype(), src0->dtype(), dst->dtype(), ld_src1 / src1_blck_size, ld_src0 / src0_blck_size, ld_dst / blck_size(dst->dtype()))
        && dst->aggregatedTensors().empty()) {
        int is_0 = (src1->batch() == 1 && src1->head() == 1) ? 0 : 1;
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++) {
            for (int64_t h = 0; h < dst->head(); h++) {
                for (int id = 0; id < thread_count; id++) {
                    llamafile_sgemm(
                        use_N, M, use_K / blck_size(src0->dtype()),
                        (char *)src1->rawHostPtr()
                            + src1->offset(b * is_0, h * is_0, 0, 0) * src1_type_size
                                  / src1_blck_size,
                        ld_src1,
                        (char *)src0->rawHostPtr()
                            + src0->offset(b, h, 0, 0) * src0_type_size / src0_blck_size,
                        ld_src0,
                        (char *)dst->rawHostPtr()
                            + dst->offset(b, h, 0, 0) * type_size(dst->dtype())
                                  / blck_size(dst->dtype()),
                        ld_dst, id, thread_count, src1->dtype(), src0->dtype(), dst->dtype());
                }
            }
        }
        return MLLM_NO_ERROR;
    }
#endif
    auto not_vec_dot_type = src0_dtype != vec_dot_type;
    std::unique_ptr<Tensor> to; // later this tensor will be freed by ~Tensor
    if (not_vec_dot_type) {
        // convert x.dtype to vec_dot_type
        // so that we can use vec_dot to calculate dot product
        assert(src0_dtype == MLLM_TYPE_F32); // x should be fp32
        to = std::make_unique<Tensor>(src0->shape());
        to->setBackend(src0->backend());
        to->setDtype(vec_dot_type);
        to->alloc();
        to->setName(src0->name() + "-vec_dot");
        int64_t i_processed = 0;
        if (from_float_to_mat && gemv && dst->masterTensor() == nullptr) {
            for (int b = 0; b < src0->batch(); b++) {
                for (int h = 0; h < src0->head(); h++) {
#pragma omp parallel for collapse(1) num_threads(thread_count)
                    for (int64_t s = 0; s < src0->sequence() - src0->sequence() % 4; s += 4) {
                        from_float_to_mat(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                          (char *)to->rawHostPtr()
                                              + to->offset(b, h, s, 0) * type_size(to->dtype())
                                                    / blck_size(to->dtype()),
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
                                      (char *)to->rawHostPtr()
                                          + to->offset(b, h, s, 0) * type_size(to->dtype())
                                                / blck_size(to->dtype()),
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
    ld_src1 = src1->sequenceSkipDim();
    ld_src0 = src0->sequenceSkipDim();
    ld_dst = dst->sequenceSkipDim();
    if (check_llamafile_sgemm(N, M, K / blck_size(src1->dtype()), src1->dtype(), src0->dtype(),
                              dst->dtype(), ld_src1 / src1_blck_size, ld_src0 / src0_blck_size, ld_dst / blck_size(dst->dtype()))
        && !support_bias && dst->ctype() == BSHD && dst->aggregatedTensors().empty()) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++) {
            for (int64_t h = 0; h < dst->head(); h++) {
                for (int id = 0; id < thread_count; id++) {
                    llamafile_sgemm(
                        use_N, M, use_K / blck_size(src1->dtype()),
                        (char *)src1->rawHostPtr()
                            + src1->offset(b, h, 0, 0) * src1_type_size / src1_blck_size,
                        ld_src1 / src1_blck_size,
                        (char *)src0->rawHostPtr()
                            + src0->offset(b, h, 0, 0) * src0_type_size / src0_blck_size,
                        ld_src0 / src0_blck_size,
                        (char *)dst->rawHostPtr()
                            + dst->offset(b, h, 0, 0) * type_size(dst->dtype())
                                  / blck_size(dst->dtype()),
                        ld_dst / blck_size(dst->dtype()), id, thread_count, src1->dtype(),
                        src0->dtype(), dst->dtype());
                }
            }
        }
        if (not_vec_dot_type) to->free();
        return MLLM_NO_ERROR;
    }
#endif

    if (gemv && !support_bias) {
        int nth = thread_count;
#pragma omp parallel for collapse(1) num_threads(thread_count)
        for (int ith = 0; ith < nth; ith++) {
            int64_t i_processed = 0;
            int64_t seq_start = (ith * use_N) / nth;
            int64_t seq_end = ((ith + 1) * use_N) / nth;
            if (gemm && (M > 3) && dst->masterTensor() == nullptr) {
                gemm(use_K, dst->hostPtr<float>() + dst->offset(0, 0, 0, seq_start), use_N,
                     (char *)src1->rawHostPtr()
                         + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                     (char *)src0->rawHostPtr(), M - M % 4, use_N / nth, /*bias=*/nullptr);
                i_processed = M - M % 4;
            }
            for (int iter = i_processed; iter < M; iter++) { // M-M%4
                gemv(use_K, dst->hostPtr<float>() + dst->offset(0, 0, iter, seq_start), use_N,
                     (char *)src1->rawHostPtr()
                         + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                     (char *)src0->rawHostPtr()
                         + src0->offset(0, 0, iter, 0) * src0_type_size / src0_blck_size,
                     1, use_N / nth, /*bias=*/nullptr);
            }
        }
        if (not_vec_dot_type) to->free();
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
                        float tmp = 0;
                        vec_dot(use_K, &tmp,
                                (char *)src1_cal->rawHostPtr()
                                    + src1_cal->offset(b * is_0, h * is_0, s_1, d_1)
                                          * src1_type_size / src1_blck_size,
                                (char *)src0_cal->rawHostPtr()
                                    + src0_cal->offset(b, h, s_0, d_0) * src0_type_size
                                          / src0_blck_size);
                        if (dst->dtypeAt(b, h, m, n) == MLLM_TYPE_F32) {
                            dst->setDataAt<float>(b, h, m, n, tmp);
                            if (support_bias) {
                                *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<float>(0, 0, 0, n);
                            }
                        } else if (dst->dtypeAt(b, h, m, n) == MLLM_TYPE_F16) {
                            if (support_bias) {
                                *dst->ptrAt<mllm_fp16_t>(b, h, m, n) =
                                    MLLM_FP32_TO_FP16(tmp + bias->dataAt<float>(0, 0, 0, n));
                            } else {
                                *dst->ptrAt<mllm_fp16_t>(b, h, m, n) = MLLM_FP32_TO_FP16(tmp);
                            }
                        } else {
                            std::cout << "Not support type [Matmul]" << std::endl;
                        }
                    }
                }
            }
        }
    }
    if (not_vec_dot_type) to->free();
    return MLLM_NO_ERROR;
}