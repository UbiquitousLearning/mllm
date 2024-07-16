//
// Created by Rongjie Yi on 23-10-24.
//

#ifndef MLLM_MATMUL_HPP
#define MLLM_MATMUL_HPP


#include "VecDot.hpp"
using namespace mllm;

ErrorCode mat_mul_elastic_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false, int thread_count=4);
ErrorCode mat_mul_elastic_fp32_fp16(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false, int thread_count=4);
ErrorCode mat_mul_elastic_fp32_q4_0(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int thread_count=4);
ErrorCode mat_mul_elastic_fp32_q4_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int thread_count=4);
ErrorCode mat_mul_elastic_fp32_q6_K(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int thread_count=4);

#endif // MLLM_MATMUL_HPP
