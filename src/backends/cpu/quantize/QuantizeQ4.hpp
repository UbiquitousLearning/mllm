//
// Created by ey on 23-10-23.
//

#ifndef MLLM_QUANTIZEQ4_HPP
#define MLLM_QUANTIZEQ4_HPP

#include "Quantize.hpp"

void quantize_row_q4_0(const float * __restrict x, void * __restrict y, int k);
void dequantize_row_q4_0(const void * __restrict vx, float * __restrict y, int k);


void quantize_row_q4_K(const float * __restrict x, void * __restrict vy, int k);
void dequantize_row_q4_K(const block_q4_K * __restrict x, float * __restrict y, int k);
#endif // MLLM_QUANTIZEQ4_HPP
