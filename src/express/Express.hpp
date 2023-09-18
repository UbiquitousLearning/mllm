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

    EOP(std::string n, OpType t) : name(n), type(t) {}
};

// 定义 ETENSOR 结构体
struct ETENSOR {
    // std::string name;
    EOP op;

    ETENSOR(EOP o) :op(o) {}
    // ETENSOR(std::string n) : name(n), op(EOP("EOP0"))  {}
};


int express_test();
ETENSOR _EOP_(std::string name, OpType type,std::vector<ETENSOR> inputs);

ETENSOR _Input();
ETENSOR _Add(std::string name, std::vector<ETENSOR> inputs);
ETENSOR _CausalMask(std::string name, std::vector<ETENSOR> inputs);
ETENSOR _MatMul(std::string name, std::vector<ETENSOR> inputs);
ETENSOR _RMSNorm(std::string name, std::vector<ETENSOR> inputs);
ETENSOR _RoPE(std::string name, std::vector<ETENSOR> inputs);
ETENSOR _Scale(std::string name, std::vector<ETENSOR> inputs);
ETENSOR _SiLU(std::string name, std::vector<ETENSOR> inputs);
ETENSOR _SoftMax(std::string name, std::vector<ETENSOR> inputs);

void createNetParem(ETENSOR endT, NetParameter& net_param_) ;
#endif //MLLM_EXPRESS_H