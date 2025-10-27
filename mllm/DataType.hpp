#pragma once

#include <cstdint>

/**
 * fp 16 type
 */
#if defined(__ARM_NEON) && !defined(_MSC_VER)
typedef __fp16 mllm_fp16_t;
#else
typedef uint16_t mllm_fp16_t;
#endif

/**
 * k quantization
 */

// #define MLLM_QKK_64
#ifdef MLLM_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

/**
 * 2-bits quantization
 */
#define QK2_0 32
#pragma pack(1)
typedef struct {
    mllm_fp16_t d;         // delta
    uint8_t qs[QK2_0 / 4]; // 2-bit quants
} block_q2_0;
#pragma pack()
static_assert(sizeof(block_q2_0) == sizeof(mllm_fp16_t) + QK2_0 / 4, "wrong q2_0 block size/padding");

#pragma pack(1)
typedef struct {
    uint8_t scales[QK_K / 16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K / 4];      // quants
    mllm_fp16_t d;             // super-block scale for quantized scales
    mllm_fp16_t dmin;          // super-block scale for quantized mins
} block_q2_K;
#pragma pack()
static_assert(sizeof(block_q2_K) == 2 * sizeof(mllm_fp16_t) + QK_K / 16 + QK_K / 4, "wrong q2_K block size/padding");

#pragma pack(1)
typedef struct {
    mllm_fp16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;
#pragma pack()
static_assert(sizeof(block_iq2_xxs) == sizeof(mllm_fp16_t) + QK_K / 8 * sizeof(uint16_t), "wrong iq2_xxs block size/padding");

/**
 * 3-bits quantization
 */
#pragma pack(1)
typedef struct {
    uint8_t hmask[QK_K / 8]; // quants - high bit
    uint8_t qs[QK_K / 4];    // quants - low 2 bits
    uint8_t scales[12];      // scales, quantized with 6 bits
    mllm_fp16_t d;           // super-block scale
} block_q3_K;
#pragma pack()
static_assert(sizeof(block_q3_K) == sizeof(mllm_fp16_t) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

/**
 * 4-bits quantization
 */
#define QK4_0 32
#pragma pack(1)
typedef struct {
    mllm_fp16_t d;         // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
#pragma pack()

//  4-bit quantization; 16 blocks of 32 elements each  weight is represented as x = a * q + b; Effectively 4.5 bits per weight
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
    mllm_fp16_t d[4];      // deltas for 4 q4_0 blocks
    uint8_t qs[QK4_0 * 2]; // nibbles / quants for 4 q4_0 blocks
} block_q4_0x4;
#pragma pack()
static_assert(sizeof(block_q4_0x4) == 4 * sizeof(mllm_fp16_t) + QK4_0 * 2, "wrong q4_0x4 block size/padding");

#pragma pack(1)
typedef struct {
    mllm_fp16_t d[8];      // deltas for 8 q4_0 blocks
    uint8_t qs[QK4_0 * 4]; // nibbles / quants for 8 q4_0 blocks
} block_q4_0x8;
#pragma pack()
static_assert(sizeof(block_q4_0x8) == 8 * sizeof(mllm_fp16_t) + QK4_0 * 4, "wrong q4_0x8 block size/padding");

/**
 * 6-bits quantization
 */
#pragma pack(1)
typedef struct {
    uint8_t ql[QK_K / 2];     // quants, lower 4 bits
    uint8_t qh[QK_K / 4];     // quants, upper 2 bits
    int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
    mllm_fp16_t d;            // super-block scale
} block_q6_K;
#pragma pack()
static_assert(sizeof(block_q6_K) == sizeof(mllm_fp16_t) + QK_K / 16 + 3 * QK_K / 4, "wrong q6_K block size/padding");

/**
 * 8-bits quantization
 */
#define QK8_0 32
#pragma pack(1)
typedef struct {
    mllm_fp16_t d;    // delta
    int8_t qs[QK8_0]; // quants
} block_q8_0;
#pragma pack()

#pragma pack(1)
typedef struct {
    int8_t qs[QK8_0];  // quants
} block_q8_per_tensor; // used in vecdot_i8_i8, TODO: remove
#pragma pack()

#define QK8_0F 32
#pragma pack(1)
typedef struct {
    float scale;       // delta
    int8_t qs[QK8_0F]; // quants
} block_q8_0f;
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
    mllm_fp16_t d[4];     // deltas for 4 q8_0 blocks
    int8_t qs[QK8_0 * 4]; // quants for 4 q8_0 blocks
} block_q8_0x4;
#pragma pack()
static_assert(sizeof(block_q8_0x4) == 4 * sizeof(mllm_fp16_t) + QK8_0 * 4, "wrong q8_0x4 block size/padding");

#pragma pack(1)
typedef struct {
    mllm_fp16_t d[8];     // deltas for 8 q8_0 blocks
    int8_t qs[QK8_0 * 8]; // quants for 8 q8_0 blocks
} block_q8_0x8;
#pragma pack()
static_assert(sizeof(block_q8_0x8) == 8 * sizeof(mllm_fp16_t) + QK8_0 * 8, "wrong q8_0x8 block size/padding");
