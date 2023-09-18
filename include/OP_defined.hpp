#ifndef MLLM_OP_DEFINED
#define MLLM_OP_DEFINED


#include <string>
#include <vector>
using std::string;
using std::vector;

namespace mllm {
enum OpType{
    Input=0,
    Add,
    SoftMax,
    Silu,
    Matmul,
    Scale,
    RoPE,
    RMSNorm,
    CausalMask,
    OP_NUM
};


static const vector<string> OpNames={ 
    "Input",
    "Add",
    "SoftMax",
    "SiLU",
    "MatMul",
    "Scale",
    "RoPE",
    "RMSNorm",
    "CausalMask",
    "OP_NUM"    
    };
}
#endif
