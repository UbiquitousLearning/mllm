#ifndef MLLM_OPDEFINED_H
#define MLLM_OPDEFINED_H

#include <string>
#include <vector>
using std::string;
using std::vector;

namespace mllm {
enum OpType {
    INVALID_VALUE = 0,
    PARAMETER,
    ADD,
    SOFTMAX,
    SILU,
    MATMUL,
    SCALE,
    ROPE,
    RMSNORM,
    CAUSALMASK,
    LINEAR,
    EMBEDDING,
    MUL,
    VIEW,
    KVCACHE,
    RELU,
    RELU2,
    GELU,
    QUICKGLUE,
    LAYERNORM,
    SPLIT,
    GATHER,
    CONVOLUTION2D,
    AVGPOOL2D,
    MAXPOOL2D,
    CAT,
    TRANSPOSE,
    SUBDIM,
    DIVISION,
    NORM,
    OP_NUM
};

static const vector<string> OpNames = {
    "INVALID_VALUE",
    "Parameter",
    "Add",
    "SoftMax",
    "SiLU",
    "MatMul",
    "Scale",
    "RoPE",
    "RMSNorm",
    "CausalMask",
    "Linear",
    "Embedding",
    "Mul",
    "VIEW",
    "KVCACHE",
    "ReLU",
    "ReLUSquaredActivation",
    "GELU",
    "QuickGELU",
    "LayerNorm",
    "Split",
    "Gqther",
    "Convolution2D",
    "AvgPool2D",
    "MaxPool2D",
    "Cat",
    "Transpose",
    "SubDim",
    "Division",
    "Norm",
    "OP_NUM"};
} // namespace mllm
#endif
