
#ifndef MLLM_TYPES_H
#define MLLM_TYPES_H
#include "OpDefined.hpp"
#include <iostream>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>
#include <cassert>
using std::string;
using std::vector;
using std::map;

typedef map<std::string, float> OpParam;


// #define DEBUGSAVETENSOR
// #define DEBUGOPTIME


#define LLAMAFILE_SGEMM

typedef enum {
    MLLM_CPU,
    MLLM_OPENCL,
    MLLM_QNN
} BackendType;

enum TensorStatus {
    TENSOR_DYNAMIC,
    TENSOR_STATIC_INIT,
    TENSOR_STATIC_READY,
};

enum ErrorCode {
    MLLM_NO_ERROR = 0,
    OUT_OF_MEMORY = 1,
    NOT_SUPPORT = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION = 4,
    INVALID_VALUE = 5,
};

enum DataType {
    MLLM_TYPE_F32 = 0,
    MLLM_TYPE_F16 = 1,
    MLLM_TYPE_Q4_0 = 2,
    MLLM_TYPE_Q4_1 = 3,
    MLLM_TYPE_Q8_0 = 8,
    MLLM_TYPE_Q8_1 = 9,
    // k-quantizations
    MLLM_TYPE_Q4_K = 12,
    MLLM_TYPE_Q6_K = 14,
    MLLM_TYPE_Q8_K = 15,
    MLLM_TYPE_I8,
    MLLM_TYPE_I16,
    MLLM_TYPE_I32,
    MLLM_TYPE_Q4_0_4_4=19,
    MLLM_TYPE_Q4_0_4_8=20,
    MLLM_TYPE_Q4_0_8_8=21,
    MLLM_TYPE_Q8_0_4_4,
    MLLM_TYPE_COUNT,
};
enum ChlType {
    BSHD = 0,
    BHDS = 2,

    BCTHW = 3,
    BTHWC = 4,
    BWCTH = 5,

    SBHD = 10,
    BDHS = 11,
    BDSH = 12,
    DBHS = 13
};

inline std::map<std::vector<int>, ChlType> Chls2Type = {
    {{0, 2, 3, 1}, BDHS},
    {{0, 1, 3, 2}, BHDS},
    {{0, 2, 1, 3}, BSHD},
    {{1, 2, 0, 3}, SBHD},
    {{0, 3, 2, 1}, BDSH},
    {{1, 2, 3, 0}, DBHS},
    {{0, 1, 2, 3, 4}, BTHWC},
    {{0, 2, 3, 4, 1}, BCTHW},
    {{0, 3, 4, 1, 2}, BWCTH}};

enum TensorType {
    INPUT_TENSOR = 0,
    NORMAL_TENSOR
};

enum Chl {
    BATCH = 0,
    SEQUENCE = 1,
    HEAD = 2,
    DIMENSION = 3,

    HD = 113,   // only use for split attn.in_proj
    D_HD = 313, // only use for split attn.in_proj

    CHANNLE = 1,
    TIME = 2,
    HEIGHT = 3,
    WIDTH = 4,

    THW = 234,

};

#define ANYDIM -198098

enum PaddingType {
    SAME,
    VALID
};

enum RoPEType {
    NONE = 0,
    LLAMAROPE = 2,
    PERSIMMONROPE = 3,
    HFHUBROPE = 4,
    MLAROPE = 5,
};

/*
 * This code is based on ggml(https://github.com/ggerganov/ggml),
 * please see https://github.com/ggerganov/ggml/blob/master/src/ggml.c
 * ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
 *
 * MIT License
 * Copyright (c) 2022 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if defined(__ARM_NEON) && !defined(_MSC_VER)
typedef __fp16 mllm_fp16_t;
#else
typedef uint16_t mllm_fp16_t;
#endif

// #define MLLM_QKK_64
#ifdef MLLM_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif
#define QK4_0 32

#pragma pack(1)
typedef struct {
    mllm_fp16_t d;         // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
#pragma pack()

//  4-bit quantization
//  16 blocks of 32 elements each
//  weight is represented as x = a * q + b
//  Effectively 4.5 bits per weight
#ifdef MLLM_QKK_64
#pragma pack(1)
typedef struct {
    mllm_fp16_t d[2];     // super-block scales/mins
    uint8_t scales[2];    // 4-bit block scales/mins
    uint8_t qs[QK_K / 2]; // 4--bit quants
} block_q4_K;
#pragma pack()
static_assert(sizeof(block_q4_K) == 2 * sizeof(uint16_t) + QK_K / 2 + 2, "wrong q4_K block size/padding");
#else
#pragma pack(1)
typedef struct {
    mllm_fp16_t d;                // super-block scale for quantized scales
    mllm_fp16_t dmin;             // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];         // 4--bit quants
} block_q4_K;
#pragma pack()
static_assert(sizeof(block_q4_K) == 2 * sizeof(mllm_fp16_t) + K_SCALE_SIZE + QK_K / 2, "wrong q4_K block size/padding");
#endif

#pragma pack(1)
typedef struct {
    uint8_t ql[QK_K / 2];     // quants, lower 4 bits
    uint8_t qh[QK_K / 4];     // quants, upper 2 bits
    int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
    mllm_fp16_t d;            // super-block scale
} block_q6_K;
#pragma pack()
static_assert(sizeof(block_q6_K) == sizeof(mllm_fp16_t) + QK_K / 16 + 3 * QK_K / 4, "wrong q6_K block size/padding");

#define QK8_0 32
#pragma pack(1)
typedef struct {
    mllm_fp16_t d;    // delta
    int8_t qs[QK8_0]; // quants
} block_q8_0;
#pragma pack()

// This is only used for intermediate quantization and dot products
#pragma pack(1)
typedef struct {
    float d;                  // delta
    int8_t qs[QK_K];          // quants
    int16_t bsums[QK_K / 16]; // sum of quants in groups of 16
} block_q8_K;
#pragma pack()
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K / 16 * sizeof(int16_t), "wrong q8_K block size/padding");


#pragma pack(1)
typedef struct {
    mllm_fp16_t d[4];        // deltas for 4 q4_0 blocks
    uint8_t qs[QK4_0 * 2]; // nibbles / quants for 4 q4_0 blocks
} block_q4_0x4;
#pragma pack()
static_assert(sizeof(block_q4_0x4) == 4 * sizeof(mllm_fp16_t) + QK4_0 * 2, "wrong q4_0x4 block size/padding");

#pragma pack(1)
typedef struct {
    mllm_fp16_t d[8];        // deltas for 8 q4_0 blocks
    uint8_t qs[QK4_0 * 4]; // nibbles / quants for 8 q4_0 blocks
} block_q4_0x8;
#pragma pack()
static_assert(sizeof(block_q4_0x8) == 8 * sizeof(mllm_fp16_t) + QK4_0 * 4, "wrong q4_0x8 block size/padding");

#pragma pack(1)
typedef struct {
    mllm_fp16_t d[4];        // deltas for 4 q8_0 blocks
    int8_t qs[QK8_0 * 4];  // quants for 4 q8_0 blocks
} block_q8_0x4;
#pragma pack()
static_assert(sizeof(block_q8_0x4) == 4 * sizeof(mllm_fp16_t) + QK8_0 * 4, "wrong q8_0x4 block size/padding");

#pragma pack(1)
typedef struct {
    mllm_fp16_t d[8];        // deltas for 8 q8_0 blocks
    int8_t qs[QK8_0 * 8];  // quants for 8 q8_0 blocks
} block_q8_0x8;
#pragma pack()
static_assert(sizeof(block_q8_0x8) == 8 * sizeof(mllm_fp16_t) + QK8_0 * 8, "wrong q8_0x8 block size/padding");

//

static string DataTypeName(DataType dataType) {
    switch (dataType) {
    case MLLM_TYPE_F32:
        return "F32";
    case MLLM_TYPE_F16:
        return "F16";
    case MLLM_TYPE_I32:
        return "I32";
    case MLLM_TYPE_I16:
        return "I16";
    case MLLM_TYPE_I8:
        return "I8";
    case MLLM_TYPE_Q4_0:
        return "Q4_0";
    case MLLM_TYPE_Q4_K:
        return "Q4_K";
    case MLLM_TYPE_Q6_K:
        return "Q6_K";
    case MLLM_TYPE_Q8_0:
        return "Q8_0";
    case MLLM_TYPE_Q8_K:
        return "Q8_K";
    case MLLM_TYPE_Q4_1:
        return "Q4_1";
    case MLLM_TYPE_Q8_1:
        return "Q8_1";
    case MLLM_TYPE_Q4_0_4_4:
        return "Q4_0_4_4";
    case MLLM_TYPE_Q4_0_4_8:
        return "Q4_0_4_8";
    case MLLM_TYPE_Q4_0_8_8:
        return "Q4_0_8_8";
    case MLLM_TYPE_Q8_0_4_4:
        return "Q8_0_4_4";
    case MLLM_TYPE_COUNT:
        return "COUNT";
    default:
        return "Unknown";
    }
}
static size_t DataTypeSize(DataType dtype, int count = 1) {
    switch (dtype) {
    case MLLM_TYPE_F32:
        return sizeof(float) * count;
    case MLLM_TYPE_F16:
        return sizeof(mllm_fp16_t) * count;
    case MLLM_TYPE_I32:
        return sizeof(int) * count;
    case MLLM_TYPE_I16:
        return sizeof(short) * count;
    case MLLM_TYPE_I8:
        return sizeof(char) * count;
    case MLLM_TYPE_Q4_0:
        return (sizeof(block_q4_0)) * count / (QK4_0);
    case MLLM_TYPE_Q4_K:
        return (sizeof(block_q4_K)) * count / (QK_K);
    case MLLM_TYPE_Q6_K:
        return (sizeof(block_q6_K)) * count / (QK_K);
    case MLLM_TYPE_Q8_0:
        return (sizeof(block_q8_0)) * count / (QK8_0);
    case MLLM_TYPE_Q8_K:
        return (sizeof(block_q8_K)) * count / (QK_K);
    case MLLM_TYPE_Q4_1:
    case MLLM_TYPE_Q8_1:
        return -1;
    case MLLM_TYPE_Q4_0_4_4:
        return (sizeof(block_q4_0x4)) * count / (QK4_0 * 4);
    case MLLM_TYPE_Q4_0_4_8:
        return (sizeof(block_q4_0x8)) * count / (QK4_0 * 8);
    case MLLM_TYPE_Q4_0_8_8:
        return (sizeof(block_q4_0x8)) * count / (QK4_0 * 8);
    case MLLM_TYPE_Q8_0_4_4:
        return (sizeof(block_q8_0x4)) * count / (QK8_0 * 4);
    case MLLM_TYPE_COUNT:
        return 0;
    default:
        return 0;
    }
}
#ifdef __cplusplus
namespace mllm {
// TODO: copy from MNN; need to recode
struct BackendConfig {
    enum MemoryMode {
        Memory_Normal = 0,
        Memory_High,
        Memory_Low
    };

    MemoryMode memory = Memory_Normal;

    enum PowerMode {
        Power_Normal = 0,
        Power_High,
        Power_Low
    };

    PowerMode power = Power_Normal;

    enum PrecisionMode {
        Precision_Normal = 0,
        Precision_High,
        Precision_Low
    };

    PrecisionMode precision = Precision_Normal;

    /** user defined context */
    void *sharedContext = nullptr;
};

} // namespace mllm
#endif //__cplusplus
#endif // MLLM_TYPES_H
