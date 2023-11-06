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
    MASK,
    LINEAR,
    ATTENTION,
    EMBEDDING,
    MUL,
    VIEW,
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
    "Mask",
    "Linear",
    "Attention",
    "Embedding",
    "Mul",
    "VIEW",
    "OP_NUM"};
} // namespace mllm
#endif
