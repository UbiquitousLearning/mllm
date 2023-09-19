#ifndef MLLM_EXPRESS_H
#define MLLM_EXPRESS_H

#include "NetParameter.hpp"

using namespace mllm;
// 前置声明
struct ETENSOR;

// 定义 EOP 结构体
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

// int express_test();
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

void createNetParem(ETENSOR endT, NetParameter &net_param_);
#endif // MLLM_EXPRESS_H