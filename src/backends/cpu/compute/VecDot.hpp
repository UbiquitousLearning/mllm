//
// Created by ey on 23-10-30.
//

#ifndef MLLM_VECDOT_HPP
#define MLLM_VECDOT_HPP
#include "Neon.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <functional>
#include "ParamLoader.hpp"
#include "../quantize/QuantizeQ8.hpp"
#include "../quantize/QuantizeQ4.hpp"
using namespace mllm;

void vec_dot_fp32(const float * __restrict src0, const float * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf);
void vec_dot_q4_0_q8_0(const void * __restrict src0, const void * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf);
void vec_dot_q4_K_q8_K(const void * __restrict src0, const void * __restrict src1, Tensor *dst, bool support_bias, Tensor *bias, int hid_len, int batch, int head, int src0_inf, int sec1_outf);


#endif // MLLM_VECDOT_HPP
