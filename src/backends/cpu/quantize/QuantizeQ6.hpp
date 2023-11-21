//
// Created by ey on 23-11-20.
//

#ifndef MLLM_QUANTIZEQ6_HPP
#define MLLM_QUANTIZEQ6_HPP

#include "Quantize.hpp"

void quantize_row_q6_K(const float * __restrict x, void * __restrict y, int k);
void dequantize_row_q6_K(const block_q8_K * __restrict x, float * __restrict y, int k);

#endif // MLLM_QUANTIZEQ6_HPP
