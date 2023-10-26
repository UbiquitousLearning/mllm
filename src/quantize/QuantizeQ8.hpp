//
// Created by ey on 23-10-26.
//

#ifndef MLLM_QUANTIZEQ8_HPP
#define MLLM_QUANTIZEQ8_HPP
#include "stdint.h"
#include "assert.h"
#include "math.h"
// TODO: better arch define macro
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif

#define QK8_0 32
typedef struct {
    float d;         // delta
    int8_t  qs[QK8_0];     // quants
} block_q8_0;
#endif // MLLM_QUANTIZEQ8_HPP
