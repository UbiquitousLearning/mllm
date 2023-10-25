
#ifndef MLLM_TYPES_H
#define MLLM_TYPES_H
#include "OpDefined.hpp"

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

enum mllm_dtype {
    MLLM_TYPE_F32  = 0,
    MLLM_TYPE_F16  = 1,
    MLLM_TYPE_Q4_0 = 2,
    MLLM_TYPE_Q4_1 = 3,
    // MLLM_TYPE_Q4_2 = 4, support has been removed
    // MLLM_TYPE_Q4_3 (5) support has been removed
    MLLM_TYPE_Q5_0 = 6,
    MLLM_TYPE_Q5_1 = 7,
    MLLM_TYPE_Q8_0 = 8,
    MLLM_TYPE_Q8_1 = 9,
    // k-quantizations
    MLLM_TYPE_Q2_K = 10,
    MLLM_TYPE_Q3_K = 11,
    MLLM_TYPE_Q4_K = 12,
    MLLM_TYPE_Q5_K = 13,
    MLLM_TYPE_Q6_K = 14,
    MLLM_TYPE_Q8_K = 15,
    MLLM_TYPE_I8,
    MLLM_TYPE_I16,
    MLLM_TYPE_I32,
    MLLM_TYPE_COUNT,
};

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
enum DataType {
    FP32 = 0,
    FP16,
};

} // namespace mllm
#endif //__cplusplus
#endif // MLLM_TYPES_H