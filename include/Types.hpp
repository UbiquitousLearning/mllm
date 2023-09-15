
#ifndef MLLM_TYPES_H
#define MLLM_TYPES_H
typedef enum {
    mllm_CPU,
    mllm_OPENCL,
    mllm_NNAPI
}BackendType;


enum ErrorCode {
    NO_ERROR           = 0,
    OUT_OF_MEMORY      = 1,
    NOT_SUPPORT        = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION       = 4,
    INVALID_VALUE      = 5,

    // // User error
    // INPUT_DATA_ERROR = 10,
    // CALL_BACK_STOP   = 11,

    // // Op Resize Error
    // TENSOR_NOT_SUPPORT = 20,
    // TENSOR_NEED_DIVIDE = 21,
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
    void* sharedContext = nullptr;
    };


    // 定义枚举类型
    enum OpType {
        ADD,
        MATMUL
    };
}
#endif //__cplusplus
#endif //MLLM_TYPES_H