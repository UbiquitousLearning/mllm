//
// Created by ey on 23-10-23.
//

#ifndef MLLM_QUANTIZEQ4_HPP
#define MLLM_QUANTIZEQ4_HPP

#include "Quantize.hpp"

#define QK4_0 32
//typedef uint16_t mllm_fp16_t;
typedef struct {
    uint16_t d;         // delta
//    float d;         // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;


void quantize_row_q4_0(const float * __restrict x, void * __restrict y, int k);
void dequantize_row_q4_0(const void * __restrict vx, float * __restrict y, int k);

#endif // MLLM_QUANTIZEQ4_HPP
