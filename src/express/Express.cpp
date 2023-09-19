#include <iostream>
#include <string>
#include <vector>
#include "Express.hpp"
#include <algorithm> // 包含 reverse 函数的头文件
#include <iostream>

static int op_idx = 0;

// 函数实现 _EOP_
ETENSOR _EOP_(std::string name, OpType type, std::vector<ETENSOR> inputs, OpParam op_param) {
    EOP op(name, type, op_param);
    for (auto &input : inputs)
        op.connectedETensors.push_back(input);
    return ETENSOR(op);
}

// 函数实现 createNetParem
void loopNetParem(ETENSOR end_t, NetParameter &net_param) {
    NetOp net_op_;
    // std::cout <<end_t.op.name<<": ";// << ", Connected ETensors: ";
    net_op_.name = end_t.op.name;
    net_op_.type = end_t.op.type;
    net_op_.param = end_t.op.op_param;
    for (const auto &ETensor : end_t.op.connectedETensors) {
        // std::cout << ""<<ETensor.op.name << "&";
        net_op_.in_op.push_back(ETensor.op.name);
    }
    // std::cout << std::endl;
    if (net_op_.in_op.size()) {
        net_param.net_ops.push_back(net_op_);
    }

    // 递归调用，继续打印连接的 op
    for (const auto &ETENSOR : end_t.op.connectedETensors) {
        loopNetParem(ETENSOR, net_param);
    }
}
void createNetParem(ETENSOR end_t, NetParameter &net_param) {
    loopNetParem(end_t, net_param);
    std::reverse(net_param.net_ops.begin(), net_param.net_ops.end());
}

ETENSOR _Input(vector<int> shape) {
    OpParam opparam;
    opparam["type"] = INPUT;
    ETENSOR x(EOP(OpNames[INPUT] + std::to_string(op_idx), INPUT, opparam));
    return x;
}
ETENSOR _Add(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = ADD;
    return _EOP_(OpNames[ADD] + std::to_string(op_idx), ADD, inputs, opparam);
}
ETENSOR _CausalMask(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = CAUSALMASK;
    return _EOP_(OpNames[CAUSALMASK] + std::to_string(op_idx), CAUSALMASK, inputs, opparam);
}
ETENSOR _MatMul(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = MAUMUL;
    return _EOP_(OpNames[MAUMUL] + std::to_string(op_idx), MAUMUL, inputs, opparam);
}
ETENSOR _RMSNorm(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = RMSNORM;
    return _EOP_(OpNames[RMSNORM] + std::to_string(op_idx), RMSNORM, inputs, opparam);
}
ETENSOR _RoPE(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = ROPE;
    return _EOP_(OpNames[ROPE] + std::to_string(op_idx), ROPE, inputs, opparam);
}
ETENSOR _Scale(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = SCALE;
    return _EOP_(OpNames[SCALE] + std::to_string(op_idx), SCALE, inputs, opparam);
}
ETENSOR _SiLU(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = SILU;
    return _EOP_(OpNames[SILU] + std::to_string(op_idx), SILU, inputs, opparam);
}
ETENSOR _SoftMax(std::vector<ETENSOR> inputs, int axis) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = SOFTMAX;
    opparam["axis"] = axis;
    return _EOP_(OpNames[SOFTMAX] + std::to_string(op_idx), SOFTMAX, inputs, opparam);
}