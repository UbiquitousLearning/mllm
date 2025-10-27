#ifndef MLLM_OPDEFINED_H
#define MLLM_OPDEFINED_H

#include <string>
#include <vector>
using std::string;
using std::vector;

namespace mllm {
enum OpType {
    INVALID_VALUE = 0,    // 0
    PARAMETER,            // 1
    ADD,                  // 2
    SOFTMAX,              // 3
    SILU,                 // 4
    SILU_FULL_PRECISION,  // 5
    MATMUL,               // 6
    SCALE,                // 7
    ROPE,                 // 8
    ROPESIMPLE,           // 9
    POSITIOANL_EMBEDDING, // 10
    RMSNORM,              // 11
    CAUSALMASK,           // 12
    SLIDINGWINDOWMASK,    // 13
    LINEAR,               // 14
    LINEARINT8,           // 15
    LINEARINT8SHADOW,     // 16
    EMBEDDING,            // 17
    MUL,                  // 18
    VIEW,                 // 19
    KVCACHE,              // 20
    KVCACHENPU,           // 21
    RELU,                 // 22
    RELU2,                // 23
    OP_GELU,              // 24
    QUICKGLUE,            // 25
    LAYERNORM,            // 26
    SPLIT,                // 27
    GATHER,               // 28
    CONVOLUTION2D,        // 29
    CONVOLUTION3D,        // 30
    VISIONROPE,           // 31
    VISIONROPESIN,        // 32
    VISIONROPECOS,        // 33
    MULTIMODALROPEPIP,    // 34
    MULTIMODALROPE,       // 35
    AVGPOOL2D,            // 36
    MAXPOOL2D,            // 37
    CAT,                  // 38
    TRANSPOSE,            // 39
    SUBDIM,               // 40
    DIVISION,             // 41
    NORM,                 // 42
    SHAPE,                // 43
    MEAN,                 // 44
    RANGE,                // 45
    WHERE,                // 46
    REPLACE,              // 47
    PREDICTOR,            // 48
    SPARSELINEAR,         // 49
    SPARSEIDLINEAR,       // 50
    ELASTICLINEAR,        // 51
    POSITION,             // 52
    WNOP,                 // 53
    QUANTIZE,             // 54
    DEQUANTIZE,           // 55
    DEQUANTIZEADD,        // 56
    MERGEOUTPUT,          // 57
    SPLITINPUT,           // 58
    IROPE,                // 59
    OP_NUM,               // 60
    NTKROPE,              // 61
    SCATTER,              // 62
    TILDE,                // 63
    MASKEDFILL,           // 64
    SIGMOID,              // 65

    // add in xnnpack
    DIRECT,           // 66
    DISPATCH,         // 67
    SUBGRAPHSTART,    // 68
    SUBGRAPHFINALIZE, // 69
    D2H,              // 70
    XP_KVCACHE,       // 71
    SDPA,             // 72

    // new front-end
    SUPERSILU,  // 73
    HEADLINEAR, // 74

    // for speculative decoding
    ROPETREE,       // 75
    CAUSALTREEMASK, // 76
    KVCACHESAGE,    // 77

    //
    F_ADD,             // 78
    F_SUB,             // 79
    F_MUL,             // 80
    F_DIV,             // 81
    F_DIVINT,          // 82
    F_TTADD,           // 83
    F_TTSUB,           // 84
    F_TTMUL,           // 85
    F_TTDIV,           // 86
    F_MM,              // 87
    F_NORM,            // 88
    F_MEAN,            // 89
    F_CAT,             // 90
    F_VIEW,            // 91
    F_TRANPOSE,        // 92
    F_FLATTEN,         // 93
    F_CLIP,            // 94
    F_CLIPAXIS,        // 95
    F_CLIPTENSOR,      // 96
    F_RANGE,           // 97
    F_WHERE,           // 98
    F_INDEX_PUT,       // 99
    F_SPLIT,           // 100
    F_SUM,             // 101
    F_TOPK,            // 102
    F_EXPPAND,         // 103
    F_ARGSORT,         // 104
    F_BINCOUNT,        // 105
    F_REPEAT,          // 106
    F_LIKE,            // 107
    F_SCATTERRADD,     // 108
    F_APPLY_VISIOROPE, // 109
    F_FA2,             // 110
    F_SAGEATTN,        // 111
    // models use only
    F_FUYU_GATHER_EMBD, // 112
    F_PHI3V_HD_MERGE,   // 113
};

enum TensorFuncType {
    FUNC_ADD,
    FUNC_SUB,
    FUNC_MUL,
    FUNC_DIV,
    FUNC_DIVINT,
    FUNC_TTADD,
    FUNC_TTSUB,
    FUNC_TTMUL,
    FUNC_TTDIV,
    FUNC_MM,
    FUNC_NORM,
    FUNC_MEAN,
    FUNC_CAT,
    FUNC_VIEW,
    FUNC_TRANPOSE,
    FUNC_FLATTEN,
    FUNC_CLIP,
    FUNC_CLIPAXIS,
    FUNC_CLIPTENSOR,
    FUNC_RANGE,
    FUNC_WHERE,
    FUNC_INDEX_PUT,
    FUNC_SPLIT,
    FUNC_SUM,
    FUNC_TOPK,
    FUNC_EXPPAND,
    FUNC_ARGSORT,
    FUNC_BINCOUNT,
    FUNC_REPEAT,
    FUNC_LIKE,
    FUNC_SCATTERREDUCE,
    FUNC_APPLY_VISIOROPE,
    FUNC_FA2,
    // models use only
    FUNC_FUYU_GATHER_EMBD,
    FUNC_PHI3V_HD_MERGE,
};

} // namespace mllm
#endif
