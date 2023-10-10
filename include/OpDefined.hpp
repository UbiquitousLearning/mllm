#ifndef MLLM_OPDEFINED_H
#define MLLM_OPDEFINED_H

#include <string>
#include <vector>
using std::string;
using std::vector;

namespace mllm {
enum OpType {
    INVALID_VALUE = 0,
    ADD,
    SOFTMAX,
    SILU,
    MATMUL,
    SCALE,
    ROPE,
    RMSNORM,
    CAUSALMASK,
    LINEAR,
    ATTENTION,
    EMBEDDING,
    OP_NUM
};

static const vector<string> OpNames = {
    "INVALID_VALUE",
    "Add",
    "SoftMax",
    "SiLU",
    "MatMul",
    "Scale",
    "RoPE",
    "RMSNorm",
    "CausalMask",
    "Linear",
    "Attention",
    "Embedding",
    "OP_NUM"};
} // namespace mllm
#endif
