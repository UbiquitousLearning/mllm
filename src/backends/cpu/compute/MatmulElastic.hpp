//
// Created by Rongjie Yi on 23-10-24.
//

#ifndef MLLM_MATMULELASTIC_HPP
#define MLLM_MATMULELASTIC_HPP

#include "VecDot.hpp"
using namespace mllm;

ErrorCode mat_mul_elastic(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int activate_input_dim = -1, int activate_output_dim = -1, bool transpose0 = false, bool transpose1 = true, int thread_count = 4);

#endif // MLLM_MATMULELASTIC_HPP
