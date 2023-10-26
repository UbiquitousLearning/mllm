//
// Created by ey on 23-10-24.
//

#ifndef MLLM_QUANTIZE_HPP
#define MLLM_QUANTIZE_HPP

//#include "QuantizeQ4.hpp"
//#include "QuantizeQ8.hpp"

#include "stdint.h"
#include "assert.h"
#include "math.h"
#include<string.h>
// TODO: better arch define macro
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif


#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static float table_f32_f16[1 << 16];
inline static float lookup_fp16_to_fp32(uint16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return table_f32_f16[s];
}

#define MLLM_FP16_TO_FP32(x) lookup_fp16_to_fp32(x)
#define MLLM_FP32_TO_FP16(x)  _cvtss_sh(x, 0)


#endif // MLLM_QUANTIZE_HPP
