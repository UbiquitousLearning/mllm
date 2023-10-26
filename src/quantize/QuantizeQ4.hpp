//
// Created by ey on 23-10-23.
//

#ifndef MLLM_QUANTIZEQ4_HPP
#define MLLM_QUANTIZEQ4_HPP

#include "stdint.h"
#include "assert.h"
#include "math.h"
// TODO: better arch define macro
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif


#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define QK4_0 32
//typedef uint16_t mllm_fp16_t;
typedef struct {
//    mllm_fp16_t d;         // delta
    float d;         // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;


void quantize_row_q4_0_reference(const float *x, block_q4_0 *y, int k);
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int k);

#endif // MLLM_QUANTIZEQ4_HPP
