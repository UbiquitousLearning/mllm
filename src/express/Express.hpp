#ifndef MLLM_EXPRESS_H
#define MLLM_EXPRESS_H

#include "NetParameter.hpp"
#include "Types.hpp"
#include <string>
#include <vector>

using namespace mllm;
// 前置声明
struct ETENSOR;
struct Context {
    vector<NetParameter *> sub_param_;
    vector<NetOp *> net_ops;
    std::set<NetTensor *> net_tensors;
    int idx = 0;
    int active_sub = 0;
};

void Subgraph_begin(Context *ctx);
NetTensor *_Input(Context *ctx, vector<int> dims, string name = "", DataType type = Float32);
NetTensor *_Add(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_SiLU(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Softmax(Context *ctx, std::vector<NetTensor *> inputs, int axis, string name = "");
NetTensor *_Matmul(Context *ctx, std::vector<NetTensor *> inputs, string name = "");

void Display(Context *c);

/*
// OLD VERSION
struct EOP {
    std::string name;
    std::vector<ETENSOR> connectedETensors;
    OpType type;
    OpParam op_param;

    EOP(std::string n, OpType t, OpParam op) :
        name(n), type(t), op_param(op) {
    }
};

// 定义 ETENSOR 结构体
struct ETENSOR {
    // std::string name;
    EOP op;

    ETENSOR(EOP o) :
        op(o) {
    }
    // ETENSOR(std::string n) : name(n), op(EOP("EOP0"))  {}
};
ETENSOR _EOP_(std::string name, OpType type, std::vector<ETENSOR> inputs, OpParam op_param);

ETENSOR _Input(vector<int> shape);
ETENSOR _Add(std::vector<ETENSOR> inputs);
ETENSOR _CausalMask(std::vector<ETENSOR> inputs);
ETENSOR _MatMul(std::vector<ETENSOR> inputs);
ETENSOR _RMSNorm(std::vector<ETENSOR> inputs);
ETENSOR _RoPE(std::vector<ETENSOR> inputs);
ETENSOR _Scale(std::vector<ETENSOR> inputs);
ETENSOR _SiLU(std::vector<ETENSOR> inputs);
ETENSOR _SoftMax(std::vector<ETENSOR> inputs, int axis);

void createNetParem(ETENSOR end_t, NetParameter &net_param);
*/
#endif // MLLM_EXPRESS_H