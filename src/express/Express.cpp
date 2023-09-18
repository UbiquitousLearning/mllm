#include <iostream>
#include <string>
#include <vector>
#include "Express.hpp"



// 函数实现 _EOP_
ETENSOR _EOP_(std::string name, OpType type,std::vector<ETENSOR> inputs) {
    EOP op(name, type);
    for (auto& input : inputs)
        op.connectedETensors.push_back(input);
    return ETENSOR( op);
}

// 函数实现 createNetParem
void createNetParem(ETENSOR endT, NetParameter& net_param_) {
    NetOp net_op_;
    // std::cout <<endT.op.name<<": ";// << ", Connected ETensors: ";
    net_op_.name = endT.op.name;
    net_op_.type = endT.op.type;
    for (const auto& ETensor : endT.op.connectedETensors) {
        // std::cout << ""<<ETensor.op.name << "&";
        net_op_.inOp.push_back(ETensor.op.name);
    }
    // std::cout << std::endl;
    if(net_op_.inOp.size()){
        net_param_.net_ops.push_back(net_op_);
    }
    

    // 递归调用，继续打印连接的 op
    for (const auto& ETENSOR : endT.op.connectedETensors) {
        createNetParem(ETENSOR, net_param_);
    }
}

int express_test() {
    ETENSOR x(EOP("input1", Input));
    // ETENSOR x("ETENSOR0");
    auto y = _EOP_("silu1", Silu,{x});
    x = _EOP_("matmul1", Matmul, {x, y});
    x = _EOP_("scale1", Scale,{x});

    // 输出连接的 EOP
    NetParameter net_param_;
    createNetParem(x, net_param_);


    return 0;
}


ETENSOR _Input(){
    ETENSOR x(EOP("input1", Input));
    return x;
}
ETENSOR _Add(std::string name, std::vector<ETENSOR> inputs){
    return _EOP_(name, Add, inputs);
}
ETENSOR _CausalMask(std::string name, std::vector<ETENSOR> inputs){
    return _EOP_(name, CausalMask, inputs);
}
ETENSOR _MatMul(std::string name, std::vector<ETENSOR> inputs){
    return _EOP_(name, Matmul, inputs);
}
ETENSOR _RMSNorm(std::string name, std::vector<ETENSOR> inputs){
    return _EOP_(name, RMSNorm, inputs);
}
ETENSOR _RoPE(std::string name, std::vector<ETENSOR> inputs){
    return _EOP_(name, RoPE, inputs);
}
ETENSOR _Scale(std::string name, std::vector<ETENSOR> inputs){
    return _EOP_(name, Scale, inputs);
}
ETENSOR _SiLU(std::string name, std::vector<ETENSOR> inputs){
    return _EOP_(name, Silu, inputs);
}
ETENSOR _SoftMax(std::string name, std::vector<ETENSOR> inputs){
    return _EOP_(name, SoftMax, inputs);
}