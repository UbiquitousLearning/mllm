//
// Created by ey on 23-10-26.
//

#ifndef MLLM_QUANTIZEQ8_HPP
#define MLLM_QUANTIZEQ8_HPP
#include "Quantize.hpp"

#define QK8_0 32
typedef struct {
    uint16_t d;         // delta
//    float d;         // delta
    int8_t  qs[QK8_0];     // quants
} block_q8_0;


void quantize_row_q8_0(const float * __restrict x, void * __restrict y, int k);
void dequantize_row_q8_0(const void * __restrict vx, float * __restrict y, int k);

#endif // MLLM_QUANTIZEQ8_HPP
