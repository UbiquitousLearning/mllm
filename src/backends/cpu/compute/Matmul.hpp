//
// Created by ey on 23-10-24.
//

#ifndef MLLM_MATMUL_HPP
#define MLLM_MATMUL_HPP


#include "VecDot.hpp"
using namespace mllm;

ErrorCode mat_mul_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false);
ErrorCode mat_mul_fp32_q4_0(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false);
ErrorCode mat_mul_fp32_q4_K(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0= false, bool transpose1= false);

#endif // MLLM_MATMUL_HPP
