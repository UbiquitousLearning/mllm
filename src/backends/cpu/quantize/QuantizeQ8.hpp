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

// This is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

void quantize_row_q8_K(const float * __restrict x, void * __restrict y, int k);
void dequantize_row_q8_K(const block_q8_K * __restrict x, float * __restrict y, int k);

#endif // MLLM_QUANTIZEQ8_HPP
