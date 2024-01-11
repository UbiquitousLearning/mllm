// This code is based on ggml(https://github.com/ggerganov/ggml), please see
// https://github.com/ggerganov/ggml/blob/master/src/ggml.c
// ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov,
// please see [ggml License](./GGML_LICENSE)

#ifndef MLLM_QUANTIZEQ8_HPP
#define MLLM_QUANTIZEQ8_HPP
#include "Quantize.hpp"


void quantize_row_q8_0(const float * __restrict x, void * __restrict y, int k);
void dequantize_row_q8_0(const void * __restrict vx, float * __restrict y, int k);

void quantize_row_q8_K(const float * __restrict x, void * __restrict y, int k);
void dequantize_row_q8_K(const block_q8_K * __restrict x, float * __restrict y, int k);

#endif // MLLM_QUANTIZEQ8_HPP
