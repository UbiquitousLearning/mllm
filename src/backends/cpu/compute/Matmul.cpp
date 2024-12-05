//
// Created by Rongjie Yi on 23-10-24.
//

#include "Matmul.hpp"
#include "Types.hpp"
#include "VecDotType.hpp"
#include "SGEMM.hpp"
#include <cassert>

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <omp.h>
#endif

ErrorCode mat_mul(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias,
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

#ifdef LLAMAFILE_SGEMM
    int ld_src1 = src1->sequenceSkipDim();
    int ld_src0 = src0->sequenceSkipDim();
    int ld_dst = dst->sequenceSkipDim();
    if (check_llamafile_sgemm(N, M, K / blck_size(src0->dtype()), src1->dtype(), src0->dtype(), dst->dtype(), ld_src1 / src1_blck_size, ld_src0 / src0_blck_size, ld_dst / blck_size(dst->dtype()))
        && dst->aggregatedTensors().empty()) {
        int is_0 = (src1->batch() == 1 && src1->head() == 1 && src1->batch() != src0->batch()) ? 0 : 1;
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++) {
            for (int64_t h = 0; h < dst->head(); h++) {
                for (int id = 0; id < thread_count; id++) {
                    llamafile_sgemm(
                        N, M, K / blck_size(src0->dtype()),
                        (char *)src1->rawHostPtr()
                            + src1->offset(b * is_0, h * is_0, 0, 0) * src1_type_size
                                  / src1_blck_size,
                        ld_src1 / src1_blck_size,
                        (char *)src0->rawHostPtr()
                            + src0->offset(b, h, 0, 0) * src0_type_size / src0_blck_size,
                        ld_src0 / src0_blck_size,
                        (char *)dst->rawHostPtr()
                            + dst->offset(b, h, 0, 0) * type_size(dst->dtype())
                                  / blck_size(dst->dtype()),
                        ld_dst / blck_size(dst->dtype()), id, thread_count, src1->dtype(),
                        src0->dtype(), dst->dtype(),
                        /*bias=*/support_bias ? bias->hostPtr<float>() : nullptr,
                        /*BiasType=*/support_bias ? bias->dtype() : DataType::MLLM_TYPE_F32);
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
        if ((from_float_to_mat != nullptr) && (gemv != nullptr) && dst->masterTensor() == nullptr) {
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
        && dst->dtypeAt(0, 0, 0, 0) == MLLM_TYPE_F32 && dst->ctype() == BSHD
        && dst->aggregatedTensors().empty()) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++) {
            for (int64_t h = 0; h < dst->head(); h++) {
                for (int id = 0; id < thread_count; id++) {
                    llamafile_sgemm(
                        N, M, K / blck_size(src1->dtype()),
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
                        src0->dtype(), dst->dtype(),
                        /*bias=*/
                        support_bias ? bias->hostPtr<float>() + bias->offset(b, h, 0, 0) : nullptr,
                        /*BiasType=*/support_bias ? bias->dtype() : DataType::MLLM_TYPE_F32);
                }
            }
        }
        if (not_vec_dot_type) to->free();
        return MLLM_NO_ERROR;
    }
#endif
    if ((gemv != nullptr) && dst->dtypeAt(0, 0, 0, 0) == MLLM_TYPE_F32) {
        int nth = thread_count;
        if (!support_bias) {
#pragma omp parallel for collapse(1) num_threads(thread_count)
            for (int ith = 0; ith < nth; ith++) {
                int64_t i_processed = 0;
                int64_t seq_start = (ith * N) / nth;
                int64_t seq_end = ((ith + 1) * N) / nth;
                if ((gemm != nullptr) && (M > 3) && dst->masterTensor() == nullptr) {
                    gemm(K, dst->hostPtr<float>() + dst->offset(0, 0, 0, seq_start), N,
                         (char *)src1->rawHostPtr()
                             + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                         (char *)src0->rawHostPtr(), M - M % 4, N / nth, /*bias=*/nullptr);
                    i_processed = M - M % 4;
                }
                for (int iter = i_processed; iter < M; iter++) { // M-M%4
                    gemv(K, dst->hostPtr<float>() + dst->offset(0, 0, iter, seq_start), N,
                         (char *)src1->rawHostPtr()
                             + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                         (char *)src0->rawHostPtr()
                             + src0->offset(0, 0, iter, 0) * src0_type_size / src0_blck_size,
                         1, N / nth, /*bias=*/nullptr);
                }
            }
        } else {
#pragma omp parallel for collapse(1) num_threads(thread_count)
            for (int ith = 0; ith < nth; ith++) {
                int64_t i_processed = 0;
                int64_t seq_start = (ith * N) / nth;
                int64_t seq_end = ((ith + 1) * N) / nth;
                if ((gemm != nullptr) && (M > 3) && dst->masterTensor() == nullptr) {
                    gemm(K, dst->hostPtr<float>() + dst->offset(0, 0, 0, seq_start), N,
                         (char *)src1->rawHostPtr()
                             + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                         (char *)src0->rawHostPtr(), M - M % 4, N / nth,
                         /*bias=*/bias->hostPtr<float>()
                             + bias->offset(/*b=*/0, /*h=*/0, /*s=*/0, /*d=*/seq_start));
                    i_processed = M - M % 4;
                }
                for (int iter = i_processed; iter < M; iter++) { // M-M%4
                    gemv(K, dst->hostPtr<float>() + dst->offset(0, 0, iter, seq_start), N,
                         (char *)src1->rawHostPtr()
                             + src1->offset(0, 0, seq_start, 0) * src1_type_size / src1_blck_size,
                         (char *)src0->rawHostPtr()
                             + src0->offset(0, 0, iter, 0) * src0_type_size / src0_blck_size,
                         1, N / nth,
                         /*bias=*/bias->hostPtr<float>()
                             + bias->offset(/*b=*/0, /*h=*/0, /*s=*/0, /*d=*/seq_start));
                }
            }
        }
        if (not_vec_dot_type) to->free();
        return MLLM_NO_ERROR;
    }

    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    int is_0 = (src1->batch() == 1 && src1->head() == 1 && src1->batch() != src0->batch()) ? 0 : 1;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int block = 0; block < N / blck_0 + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 && n < N; n++) {
                        int s_1;
                        int d_1;
                        int s_0;
                        int d_0;
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
                        vec_dot(K, &tmp,
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

ErrorCode mat_mul_i8(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, int thread_count, float scale1, float scale2) {
    if (support_bias) {
        std::cout << "Not support bias in mat_mul_i8" << std::endl;
        abort();
    }

    if (!transpose1) {
        std::cout << "Not support transpose1==false in mat_mul_i8" << std::endl;
        abort();
    }

#ifdef __ARM_NEON
    armv8::qt8_qt8_fp32_gemm_sdot_omp(src0->rawHostPtr(), src1->rawHostPtr(), dst->rawHostPtr(), src0->sequence(), src1->sequence(), src0->dimension(),
                                      src0->i8_scale, src1->i8_scale, transpose1);
#else
    std::cout << "mat_mul_i8 is only supported in armv8.2+" << std::endl;
    abort();
#endif

    return MLLM_NO_ERROR;
}

#ifdef __ARM_NEON
namespace mllm::armv8 {

// This function is dropped !!!
void qt8_qt8_fp32_gemv(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
                       bool transpose_b) {
    if (!transpose_b) {
        // Not Supported Yet.
        return;
    }

    // I tile K with 16 element per block. the 16 element will be load to 128bit vector.
    if (K % 16) abort();

    int32_t k_blocks = K / 16;
    int32_t n_blocks = N / 4;
    int32_t n_blocks_left = N % 4;
    float scale = SA * SB;

    for (int32_t n_block = 0; n_block < n_blocks; n_block++) {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B + 4 * K * n_block;
        auto c_ptr = (float *)C;

        // final accumulattor
        // float acc_line_normal_reg[4] = {0.f, 0.f, 0.f, 0.f};
        // float32x4_t acc_line = vmovq_n_f32(0);

        // accumulator
        int16x8x2_t acc_line_0 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};
        int16x8x2_t acc_line_1 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};
        int16x8x2_t acc_line_2 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};
        int16x8x2_t acc_line_3 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};

        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);
            int8x8_t r0_l = vget_low_s8(r0);
            int8x8_t r0_h = vget_high_s8(r0);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + 16 * k_block);
            int8x8_t n0_l = vget_low_s8(n0);
            int8x8_t n0_h = vget_high_s8(n0);
            acc_line_0.val[0] = vmlal_s8(acc_line_0.val[0], n0_h, r0_h);
            acc_line_0.val[1] = vmlal_s8(acc_line_0.val[1], n0_l, r0_l);

            // load from block, n = 1
            int8x16_t n1 = vld1q_s8(b_ptr + K + 16 * k_block);
            int8x8_t n1_l = vget_low_s8(n1);
            int8x8_t n1_h = vget_high_s8(n1);
            acc_line_1.val[0] = vmlal_s8(acc_line_1.val[0], n1_h, r0_h);
            acc_line_1.val[1] = vmlal_s8(acc_line_1.val[1], n1_l, r0_l);

            // load from block, n = 2
            int8x16_t n2 = vld1q_s8(b_ptr + 2 * K + 16 * k_block);
            int8x8_t n2_l = vget_low_s8(n2);
            int8x8_t n2_h = vget_high_s8(n2);
            acc_line_2.val[0] = vmlal_s8(acc_line_2.val[0], n2_h, r0_h);
            acc_line_2.val[1] = vmlal_s8(acc_line_2.val[1], n2_l, r0_l);

            // load from block, n = 3
            int8x16_t n3 = vld1q_s8(b_ptr + 3 * K + 16 * k_block);
            int8x8_t n3_l = vget_low_s8(n3);
            int8x8_t n3_h = vget_high_s8(n3);
            acc_line_3.val[0] = vmlal_s8(acc_line_3.val[0], n3_h, r0_h);
            acc_line_3.val[1] = vmlal_s8(acc_line_3.val[1], n3_l, r0_l);
        }

        // accumulate i16 vector to single i32 value. And turn it to float.
        int32x4_t acc_line_0_i32_sum_0 = vpaddlq_s16(acc_line_0.val[0]);
        int32x4_t acc_line_0_i32_sum_1 = vpaddlq_s16(acc_line_0.val[1]);
        int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0_i32_sum_0) + vaddvq_s32(acc_line_0_i32_sum_1);
        // acc_line = vsetq_lane_f32((float)acc_line_0_i32, acc_line, 0);
        *(c_ptr + 4 * n_block + 0) = (float)acc_line_0_i32 * scale;

        int32x4_t acc_line_1_i32_sum_0 = vpaddlq_s16(acc_line_1.val[0]);
        int32x4_t acc_line_1_i32_sum_1 = vpaddlq_s16(acc_line_1.val[1]);
        int32_t acc_line_1_i32 = vaddvq_s32(acc_line_1_i32_sum_0) + vaddvq_s32(acc_line_1_i32_sum_1);
        // acc_line = vsetq_lane_f32((float)acc_line_1_i32, acc_line, 1);
        *(c_ptr + 4 * n_block + 1) = (float)acc_line_1_i32 * scale;

        int32x4_t acc_line_2_i32_sum_0 = vpaddlq_s16(acc_line_2.val[0]);
        int32x4_t acc_line_2_i32_sum_1 = vpaddlq_s16(acc_line_2.val[1]);
        int32_t acc_line_2_i32 = vaddvq_s32(acc_line_2_i32_sum_0) + vaddvq_s32(acc_line_2_i32_sum_1);
        // acc_line = vsetq_lane_f32((float)acc_line_2_i32, acc_line, 2);
        *(c_ptr + 4 * n_block + 2) = (float)acc_line_2_i32 * scale;

        int32x4_t acc_line_3_i32_sum_0 = vpaddlq_s16(acc_line_3.val[0]);
        int32x4_t acc_line_3_i32_sum_1 = vpaddlq_s16(acc_line_3.val[1]);
        int32_t acc_line_3_i32 = vaddvq_s32(acc_line_3_i32_sum_0) + vaddvq_s32(acc_line_3_i32_sum_1);
        // acc_line = vsetq_lane_f32((float)acc_line_3_i32, acc_line, 3);
        *(c_ptr + 4 * n_block + 3) = (float)acc_line_3_i32 * scale;

        // scale it.
        // acc_line = vmulq_n_f32(acc_line, scale);

        // store
        // vst1q_f32(c_ptr + 4 * n_block, acc_line);
    }

    // perform vector dot one by one.
    for (int32_t n = 0; n < n_blocks_left; n++) {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B + 4 * (N / 4) * K;
        auto c_ptr = (float *)C + 4 * (N / 4);

        int16x8x2_t acc_line_0 = {{vmovq_n_s16(0), vmovq_n_s16(0)}};

        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);
            int8x8_t r0_l = vget_low_s8(r0);
            int8x8_t r0_h = vget_high_s8(r0);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + n * K + 16 * k_block);
            int8x8_t n0_l = vget_low_s8(n0);
            int8x8_t n0_h = vget_high_s8(n0);
            acc_line_0.val[0] = vmlal_s8(acc_line_0.val[0], n0_h, r0_h);
            acc_line_0.val[1] = vmlal_s8(acc_line_0.val[1], n0_l, r0_l);
        }

        // accumulate i16 vector to single i32 value. And turn it to float.
        int32x4_t acc_line_0_i32_sum_0 = vpaddlq_s16(acc_line_0.val[0]);
        int32x4_t acc_line_0_i32_sum_1 = vpaddlq_s16(acc_line_0.val[1]);
        int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0_i32_sum_0) + vaddvq_s32(acc_line_0_i32_sum_1);

        // store
        *(c_ptr + n) = (float)acc_line_0_i32 * scale;
    }
}

void qt8_qt8_fp32_gemv_sdot(void *A, void *B, void *C, int32_t N, int32_t K, float SA, float SB,
                            bool transpose_b) {
    if (!transpose_b) {
        // Not Supported Yet.
        return;
    }

    // I tile K with 16 element per block. the 16 element will be load to 128bit vector.
    if (K % 16) abort();

    int32_t k_blocks = K / 16;
    int32_t n_blocks = N / 4;
    int32_t n_blocks_left = N % 4;
    float scale = SA * SB;

    for (int32_t n_block = 0; n_block < n_blocks; n_block++) {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B + 4 * K * n_block;
        auto c_ptr = (float *)C;

        // accumulator
        int32x4_t acc_line_0 = vmovq_n_s32(0);
        int32x4_t acc_line_1 = vmovq_n_s32(0);
        int32x4_t acc_line_2 = vmovq_n_s32(0);
        int32x4_t acc_line_3 = vmovq_n_s32(0);

        // Only support v8.2+ with dotproduct enabled.
        // Using SDOT. The throughput will increase 4 times compared to fp32 version.
        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + 16 * k_block);
            acc_line_0 = vdotq_s32(acc_line_0, r0, n0);

            // load from block, n = 1
            int8x16_t n1 = vld1q_s8(b_ptr + K + 16 * k_block);
            acc_line_1 = vdotq_s32(acc_line_1, r0, n1);

            // load from block, n = 2
            int8x16_t n2 = vld1q_s8(b_ptr + 2 * K + 16 * k_block);
            acc_line_2 = vdotq_s32(acc_line_2, r0, n2);

            // load from block, n = 3
            int8x16_t n3 = vld1q_s8(b_ptr + 3 * K + 16 * k_block);
            acc_line_3 = vdotq_s32(acc_line_3, r0, n3);
        }

        // reduce all and save to c_ptr
        int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0);
        *(c_ptr + 4 * n_block + 0) = (float)acc_line_0_i32 * scale;

        int32_t acc_line_1_i32 = vaddvq_s32(acc_line_1);
        *(c_ptr + 4 * n_block + 1) = (float)acc_line_1_i32 * scale;

        int32_t acc_line_2_i32 = vaddvq_s32(acc_line_2);
        *(c_ptr + 4 * n_block + 2) = (float)acc_line_2_i32 * scale;

        int32_t acc_line_3_i32 = vaddvq_s32(acc_line_3);
        *(c_ptr + 4 * n_block + 3) = (float)acc_line_3_i32 * scale;
    }

    // perform vector dot one by one.
    for (int32_t n = 0; n < n_blocks_left; n++) {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B + 4 * (N / 4) * K;
        auto c_ptr = (float *)C + 4 * (N / 4);

        int32x4_t acc_line_0 = vmovq_n_s32(0);

        for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
            // load from A
            int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);

            // load from block, n = 0
            int8x16_t n0 = vld1q_s8(b_ptr + n * K + 16 * k_block);
            acc_line_0 = vdotq_s32(acc_line_0, r0, n0);
        }

        // accumulate i32 vector to single i32 value. And turn it to float.
        int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0);

        // store
        *(c_ptr + n) = (float)acc_line_0_i32 * scale;
    }
}

void qt8_qt8_fp32_kernel_4x4_sdot(void *A, void *B, void *C, int32_t N, int32_t K, float SA,
                                  float SB, bool transpose_b) {
    if (!transpose_b) {
        // Not Supported Yet.
        return;
    }

    int32_t k_blocks = K / 16;
    float scale = SA * SB;

    // 4 x K, K x 4.
    if (K % 16) abort();

    auto a_ptr = (int8_t *)A;
    auto b_ptr = (int8_t *)B;
    auto c_ptr = (float *)C;

    // accumulator should contain 16 values.
    int32x4x4_t acc_line_0 = {{vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0)}};
    int32x4x4_t acc_line_1 = {{vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0)}};
    int32x4x4_t acc_line_2 = {{vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0)}};
    int32x4x4_t acc_line_3 = {{vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0), vmovq_n_s32(0)}};

    // final accumulator
    int32_t acc_0[4] = {0, 0, 0, 0};
    int32_t acc_1[4] = {0, 0, 0, 0};
    int32_t acc_2[4] = {0, 0, 0, 0};
    int32_t acc_3[4] = {0, 0, 0, 0};

    for (int k_block = 0; k_block < k_blocks; k_block++) {
        // load 4 vector from A
        // a1
        int8x16_t a0 = vld1q_s8(a_ptr + 16 * k_block);
        int8x16_t a1 = vld1q_s8(a_ptr + 16 * k_block + K);
        int8x16_t a2 = vld1q_s8(a_ptr + 16 * k_block + 2 * K);
        int8x16_t a3 = vld1q_s8(a_ptr + 16 * k_block + 3 * K);

        // load 4 vector from B
        int8x16_t b0 = vld1q_s8(b_ptr + 16 * k_block);
        int8x16_t b1 = vld1q_s8(b_ptr + 16 * k_block + K);
        int8x16_t b2 = vld1q_s8(b_ptr + 16 * k_block + 2 * K);
        int8x16_t b3 = vld1q_s8(b_ptr + 16 * k_block + 3 * K);

        acc_line_0.val[0] = vdotq_s32(acc_line_0.val[0], a0, b0);
        acc_line_0.val[1] = vdotq_s32(acc_line_0.val[1], a0, b1);
        acc_line_0.val[2] = vdotq_s32(acc_line_0.val[2], a0, b2);
        acc_line_0.val[3] = vdotq_s32(acc_line_0.val[3], a0, b3);

        acc_line_1.val[0] = vdotq_s32(acc_line_1.val[0], a1, b0);
        acc_line_1.val[1] = vdotq_s32(acc_line_1.val[1], a1, b1);
        acc_line_1.val[2] = vdotq_s32(acc_line_1.val[2], a1, b2);
        acc_line_1.val[3] = vdotq_s32(acc_line_1.val[3], a1, b3);

        acc_line_2.val[0] = vdotq_s32(acc_line_2.val[0], a2, b0);
        acc_line_2.val[1] = vdotq_s32(acc_line_2.val[1], a2, b1);
        acc_line_2.val[2] = vdotq_s32(acc_line_2.val[2], a2, b2);
        acc_line_2.val[3] = vdotq_s32(acc_line_2.val[3], a2, b3);

        acc_line_3.val[0] = vdotq_s32(acc_line_3.val[0], a3, b0);
        acc_line_3.val[1] = vdotq_s32(acc_line_3.val[1], a3, b1);
        acc_line_3.val[2] = vdotq_s32(acc_line_3.val[2], a3, b2);
        acc_line_3.val[3] = vdotq_s32(acc_line_3.val[3], a3, b3);
    }

    acc_0[0] = vaddvq_s32(acc_line_0.val[0]);
    acc_0[1] = vaddvq_s32(acc_line_0.val[1]);
    acc_0[2] = vaddvq_s32(acc_line_0.val[2]);
    acc_0[3] = vaddvq_s32(acc_line_0.val[3]);
    int32x4_t acc_0_vec_i32 = vld1q_s32(acc_0);
    float32x4_t acc_0_vec_f32 = vcvtq_f32_s32(acc_0_vec_i32);
    float32x4_t acc_0_vec_final = vmulq_n_f32(acc_0_vec_f32, scale);
    vst1q_f32(c_ptr, acc_0_vec_final);

    acc_1[0] = vaddvq_s32(acc_line_1.val[0]);
    acc_1[1] = vaddvq_s32(acc_line_1.val[1]);
    acc_1[2] = vaddvq_s32(acc_line_1.val[2]);
    acc_1[3] = vaddvq_s32(acc_line_1.val[3]);
    int32x4_t acc_1_vec_i32 = vld1q_s32(acc_1);
    float32x4_t acc_1_vec_f32 = vcvtq_f32_s32(acc_1_vec_i32);
    float32x4_t acc_1_vec_final = vmulq_n_f32(acc_1_vec_f32, scale);
    vst1q_f32(c_ptr + N, acc_1_vec_final);

    acc_2[0] = vaddvq_s32(acc_line_2.val[0]);
    acc_2[1] = vaddvq_s32(acc_line_2.val[1]);
    acc_2[2] = vaddvq_s32(acc_line_2.val[2]);
    acc_2[3] = vaddvq_s32(acc_line_2.val[3]);
    int32x4_t acc_2_vec_i32 = vld1q_s32(acc_2);
    float32x4_t acc_2_vec_f32 = vcvtq_f32_s32(acc_2_vec_i32);
    float32x4_t acc_2_vec_final = vmulq_n_f32(acc_2_vec_f32, scale);
    vst1q_f32(c_ptr + 2 * N, acc_2_vec_final);

    acc_3[0] = vaddvq_s32(acc_line_3.val[0]);
    acc_3[1] = vaddvq_s32(acc_line_3.val[1]);
    acc_3[2] = vaddvq_s32(acc_line_3.val[2]);
    acc_3[3] = vaddvq_s32(acc_line_3.val[3]);
    int32x4_t acc_3_vec_i32 = vld1q_s32(acc_3);
    float32x4_t acc_3_vec_f32 = vcvtq_f32_s32(acc_3_vec_i32);
    float32x4_t acc_3_vec_final = vmulq_n_f32(acc_3_vec_f32, scale);
    vst1q_f32(c_ptr + 3 * N, acc_3_vec_final);
}

void qt8_qt8_fp32_vec_dot(void *A, void *B, void *C, int32_t K, float SA, float SB) {
    if (K % 16) abort();

    int32_t k_blocks = K / 16;
    float scale = SA * SB;

    auto a_ptr = (int8_t *)A;
    auto b_ptr = (int8_t *)B;
    auto c_ptr = (float *)C;

    // accumulator
    int32x4_t acc_line_0 = vmovq_n_s32(0);

    // Only support v8.2+ with dotproduct enabled.
    // Using SDOT. The throughput will increase 4 times compared to fp32 version.
    for (int32_t k_block = 0; k_block < k_blocks; k_block++) {
        // load from A
        int8x16_t r0 = vld1q_s8(a_ptr + 16 * k_block);

        // load from block, n = 0
        int8x16_t n0 = vld1q_s8(b_ptr + 16 * k_block);
        acc_line_0 = vdotq_s32(acc_line_0, r0, n0);
    }

    // reduce all and save to c_ptr
    int32_t acc_line_0_i32 = vaddvq_s32(acc_line_0);
    *(c_ptr) = (float)acc_line_0_i32 * scale;
}

// This function is dropped !!!
void qt8_qt8_fp32_gemm(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                       float SB, bool transpose_b) {
    if (M == 1) {
        qt8_qt8_fp32_gemv(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        // FIXME: tile in more efficient way.
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;
        for (int m = 0; m < M; ++m) {
            qt8_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
        }
    }
}

// This function is dropped !!!
void qt8_qt8_fp32_gemm_omp(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                           float SB, bool transpose_b) {
    if (M == 1) {
        qt8_qt8_fp32_gemv(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        // FIXME: tile in more efficient way.
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;
        if (M > 64) {
#pragma omp parallel for num_threads(4)
            for (int m = 0; m < M; ++m) {
                qt8_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
        } else {
            for (int m = 0; m < M; ++m) {
                qt8_qt8_fp32_gemv(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
        }
    }
}

void qt8_qt8_fp32_gemm_sdot(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K, float SA,
                            float SB, bool transpose_b) {
    if (M == 1) {
        qt8_qt8_fp32_gemv_sdot(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;

        int32_t m_blocks = M / 4;
        int32_t n_blocks = N / 4;
        int32_t m_left = M % 4;
        int32_t n_left = N % 4;

        if (M < 4 || N < 4) {
            for (int m = 0; m < M; ++m) {
                qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
            return;
        }

        // main loop
        for (int m = 0; m < m_blocks; ++m) {
            for (int n = 0; n < n_blocks; ++n) {
                qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + 4 * n * K,
                                             c_ptr + 4 * m * N + 4 * n, N, K, SA, SB, transpose_b);
            }

            // some re cauculating may be needed
            if (n_left) {
                qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                             c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA, SB,
                                             transpose_b);
            }
        }

        if (m_left) {
            for (int m = m_blocks * 4; m < M; ++m) {
                qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
        }
    }
}

void qt8_qt8_fp32_gemm_sdot_omp(void *A, void *B, void *C, int32_t M, int32_t N, int32_t K,
                                float SA, float SB, bool transpose_b) {
    if (M == 1) {
        qt8_qt8_fp32_gemv_sdot(A, B, C, N, K, SA, SB, transpose_b);
    } else {
        auto a_ptr = (int8_t *)A;
        auto b_ptr = (int8_t *)B;
        auto c_ptr = (float *)C;

        int32_t m_blocks = M / 4;
        int32_t n_blocks = N / 4;
        int32_t m_left = M % 4;
        int32_t n_left = N % 4;

        if (M < 4 || N < 4) {
            for (int m = 0; m < M; ++m) {
                qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
            }
            return;
        }

        if (M > 64) {
            // main loop
#pragma omp parallel for num_threads(4)
            for (int m = 0; m < m_blocks; ++m) {
                for (int n = 0; n < n_blocks; ++n) {
                    qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + 4 * n * K,
                                                 c_ptr + 4 * m * N + 4 * n, N, K, SA, SB, transpose_b);
                }

                // some re cauculating may be needed
                if (n_left) {
                    qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                                 c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA,
                                                 SB, transpose_b);
                }
            }

            if (m_left) {
                for (int m = m_blocks * 4; m < M; ++m) {
                    qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
                }
            }
        } else {
            // main loop
            for (int m = 0; m < m_blocks; ++m) {
                for (int n = 0; n < n_blocks; ++n) {
                    qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + 4 * n * K,
                                                 c_ptr + 4 * m * N + 4 * n, N, K, SA, SB, transpose_b);
                }

                // some re cauculating may be needed
                if (n_left) {
                    qt8_qt8_fp32_kernel_4x4_sdot(a_ptr + 4 * m * K, b_ptr + (n_blocks * 4 - (4 - n_left)) * K,
                                                 c_ptr + 4 * m * N + n_blocks * 4 - (4 - n_left), N, K, SA,
                                                 SB, transpose_b);
                }
            }

            if (m_left) {
                for (int m = m_blocks * 4; m < M; ++m) {
                    qt8_qt8_fp32_gemv_sdot(a_ptr + m * K, b_ptr, c_ptr + m * N, N, K, SA, SB, transpose_b);
                }
            }
        }
    }
}
} // namespace mllm::armv8
#endif