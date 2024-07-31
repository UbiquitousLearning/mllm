//
// Created by Rongjie Yi on 23-10-24.
//

#ifndef MLLM_MATMUL_HPP
#define MLLM_MATMUL_HPP


#include "VecDot.hpp"
using namespace mllm;

ErrorCode sparse_mat_mul_id(Tensor *x, Tensor *W, Tensor *ids, Tensor *dst, int thread_count=4);
ErrorCode mat_mul_sparse(Tensor *x, Tensor *W, Tensor *dst, int thread_count=4);

ErrorCode mat_mul(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = true, int thread_count=4);
/*
ErrorCode mat_mul_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false, int thread_count=4);
ErrorCode mat_mul_fp32_fp16(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false, int thread_count=4);
ErrorCode mat_mul_fp32_q4_0(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int thread_count=4);
ErrorCode mat_mul_fp32_q4_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int thread_count=4);
ErrorCode mat_mul_fp32_q6_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int thread_count=4);
*/

ErrorCode mat_mul_elastic(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int activate_input_dim=-1, int activate_output_dim=-1,bool transpose0 = false, bool transpose1 = true, int thread_count=4);

// smoothquant int8 matmul
ErrorCode mat_mul_i8(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false, int thread_count = 4, float scale1 = 1.0f, float scale2 = 1.0f);
ErrorCode mat_mul_fp32_i8(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false, int thread_count = 4, float scale2 = 1.0f);
#endif // MLLM_MATMUL_HPP
