#include <iostream>
#include <string>
#include <vector>
#include "Express.hpp"
#include <algorithm>  // 包含 reverse 函数的头文件
#include <iostream>


static int op_idx = 0;

// 函数实现 _EOP_
ETENSOR _EOP_(std::string name, OpType type,std::vector<ETENSOR> inputs, OpParam op_param) {
    EOP op(name, type, op_param);
    for (auto& input : inputs)
        op.connectedETensors.push_back(input);
    return ETENSOR( op);
}

// 函数实现 createNetParem
void loopNetParem(ETENSOR endT, NetParameter& net_param_) {
    NetOp net_op_;
    // std::cout <<endT.op.name<<": ";// << ", Connected ETensors: ";
    net_op_.name = endT.op.name;
    net_op_.type = endT.op.type;
    net_op_.param = endT.op.op_param;
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
        loopNetParem(ETENSOR, net_param_);
    }
}
void createNetParem(ETENSOR endT, NetParameter& net_param_){
    loopNetParem(endT, net_param_);
    std::reverse( net_param_.net_ops.begin(),  net_param_.net_ops.end());
}

// int express_test() {
//     ETENSOR x(EOP("Input0", Input));
//     // ETENSOR x("ETENSOR0");
//     auto y = _EOP_("silu1", Silu,{x});
//     x = _EOP_("matmul1", Matmul, {x, y});
//     x = _EOP_("scale1", Scale,{x});

//     // 输出连接的 EOP
//     NetParameter net_param_;
//     createNetParem(x, net_param_);


//     return 0;
// }

ETENSOR _Input(vector<int> shape){
    OpParam opparam;
    ETENSOR x(EOP(OpNames[Input]+std::to_string(op_idx), Input, opparam));
    return x;
}
ETENSOR _Add(std::vector<ETENSOR> inputs){
    op_idx++;
    OpParam opparam;
    return _EOP_(OpNames[Add]+std::to_string(op_idx), Add, inputs, opparam);
}
ETENSOR _CausalMask(std::vector<ETENSOR> inputs){
    op_idx++;
    OpParam opparam;
    return _EOP_(OpNames[CausalMask]+std::to_string(op_idx), CausalMask, inputs, opparam);
}
ETENSOR _MatMul(std::vector<ETENSOR> inputs){
    op_idx++;
    OpParam opparam;
    return _EOP_(OpNames[Matmul]+std::to_string(op_idx), Matmul, inputs, opparam);
}
ETENSOR _RMSNorm(std::vector<ETENSOR> inputs){
    op_idx++;
    OpParam opparam;
    return _EOP_(OpNames[RMSNorm]+std::to_string(op_idx), RMSNorm, inputs, opparam);
}
ETENSOR _RoPE(std::vector<ETENSOR> inputs){
    op_idx++;
    OpParam opparam;
    return _EOP_(OpNames[RoPE]+std::to_string(op_idx), RoPE, inputs, opparam);
}
ETENSOR _Scale(std::vector<ETENSOR> inputs){
    op_idx++;
    OpParam opparam;
    return _EOP_(OpNames[Scale]+std::to_string(op_idx), Scale, inputs, opparam);
}
ETENSOR _SiLU(std::vector<ETENSOR> inputs){
    op_idx++;
    OpParam opparam;
    return _EOP_(OpNames[Silu]+std::to_string(op_idx), Silu, inputs, opparam);
}
ETENSOR _SoftMax(std::vector<ETENSOR> inputs, int axis){
    op_idx++;
    OpParam opparam;
    opparam["axis"] = axis;
    return _EOP_(OpNames[SoftMax]+std::to_string(op_idx), SoftMax, inputs, opparam);
}