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

//#define QK_K 64
// 4-bit quantization
// 16 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
#ifdef MLLM_QKK_64
typedef struct {
    uint16_t d[2];          // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(uint16_t) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
    uint16_t d;             // super-block scale for quantized scales
    uint16_t dmin;          // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(uint16_t) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");
#endif


void quantize_row_q4_K(const float * __restrict x, void * __restrict vy, int k);
void dequantize_row_q4_K(const block_q4_K * __restrict x, float * __restrict y, int k);
#endif // MLLM_QUANTIZEQ4_HPP
