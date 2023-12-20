
#ifndef MLLM_TYPES_H
#define MLLM_TYPES_H
#include "OpDefined.hpp"
#include <iostream>

typedef enum {
    MLLM_CPU,
    MLLM_OPENCL,
    MLLM_NNAPI
} BackendType;

enum ErrorCode {
    NO_ERROR = 0,
    OUT_OF_MEMORY = 1,
    NOT_SUPPORT = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION = 4,
    INVALID_VALUE = 5,

    // // User error
    // INPUT_DATA_ERROR = 10,
    // CALL_BACK_STOP   = 11,

    // // Op Resize Error
    // TENSOR_NOT_SUPPORT = 20,
    // TENSOR_NEED_DIVIDE = 21,
};

enum DataType {
    MLLM_TYPE_F32 = 0,
    MLLM_TYPE_F16 = 1,
    MLLM_TYPE_Q4_0 = 2,
    MLLM_TYPE_Q4_1 = 3,
    // MLLM_TYPE_Q4_2 = 4, support has been removed
    // MLLM_TYPE_Q4_3 (5) support has been removed
    // MLLM_TYPE_Q5_0 = 6,
    // MLLM_TYPE_Q5_1 = 7,
    MLLM_TYPE_Q8_0 = 8,
    MLLM_TYPE_Q8_1 = 9,
    // k-quantizations
    // MLLM_TYPE_Q2_K = 10,
    // MLLM_TYPE_Q3_K = 11,
    MLLM_TYPE_Q4_K = 12,
    // MLLM_TYPE_Q5_K = 13,
    MLLM_TYPE_Q6_K = 14,
    MLLM_TYPE_Q8_K = 15,
    MLLM_TYPE_I8,
    MLLM_TYPE_I16,
    MLLM_TYPE_I32,
    MLLM_TYPE_COUNT,
};
enum ChlType {
    BSHD = 0,
    // BHSD = 1, //ABANDENED!!
    BHDS = 2,
};

enum Chl {
    BATCH = 0,
    HEAD = 1,
    SEQUENCE = 2,
    DIMENSION = 3,
    D_HD,
};


enum PaddingType {
    SAME,
    VALID
};

#if defined(__ARM_NEON) && !defined(_MSC_VER)
typedef __fp16 mllm_fp16_t;
#else
typedef uint16_t mllm_fp16_t;
#endif

//#define MLLM_QKK_64
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
    mllm_fp16_t d;            // delta
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
    mllm_fp16_t d[2];        // super-block scales/mins
    uint8_t scales[2];    // 4-bit block scales/mins
    uint8_t qs[QK_K / 2]; // 4--bit quants
} block_q4_K;
#pragma pack()
static_assert(sizeof(block_q4_K) == 2 * sizeof(uint16_t) + QK_K / 2 + 2, "wrong q4_K block size/padding");
#else
#pragma pack(1)
typedef struct {
    mllm_fp16_t d;                   // super-block scale for quantized scales
    mllm_fp16_t dmin;                // super-block scale for quantized mins
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
static_assert(sizeof(block_q6_K) == sizeof(mllm_fp16_t) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

#define QK8_0 32
#pragma pack(1)
typedef struct {
    mllm_fp16_t d;       // delta
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
    case MLLM_TYPE_COUNT:
        return "COUNT";
    default:
        return "Unknown";
    }
}
static size_t DataTypeSize(DataType dtype, int count=1) {
    switch (dtype) {
    case MLLM_TYPE_F32:
        return sizeof(float) *count;
    case MLLM_TYPE_F16:
        return sizeof(mllm_fp16_t)*count;
    case MLLM_TYPE_I32:
        return sizeof(int)*count;
    case MLLM_TYPE_I16:
        return sizeof(short)*count;
    case MLLM_TYPE_I8:
        return sizeof(char)*count;
    case MLLM_TYPE_Q4_0:
        return (sizeof(block_q4_0))*count / (QK4_0);
    case MLLM_TYPE_Q4_K:
        return (sizeof(block_q4_K))*count / (QK_K);
    case MLLM_TYPE_Q6_K:
        return (sizeof(block_q6_K))*count / (QK_K);
    case MLLM_TYPE_Q8_0:
        return (sizeof(block_q8_0))*count / (QK8_0);
    case MLLM_TYPE_Q8_K:
        return (sizeof(block_q8_K))*count / (QK_K);
    case MLLM_TYPE_Q4_1:
    case MLLM_TYPE_Q8_1:
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

// 定义枚举类型
// enum DataType {
//    FP32 = 0,
//    FP16,
//    INT8,
//    INT4,
//    DATA_TYPE_COUNT,
//};

} // namespace mllm
#endif //__cplusplus
#endif // MLLM_TYPES_H