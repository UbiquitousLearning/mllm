#include <iostream>
// #include "Graph.hpp"
// #include "Tensor.hpp"
// #include "backends/cpu/CPUMatmul.hpp"
#include "Net.hpp"
#include "Executor.hpp"


#include "express/Express.hpp"

using namespace mllm;

int main()
{
    // Tensor tensor_(1,3,5,5);
    // std::cout << "Init Tensor" << std::endl;
    // tensor_.Reshape(1,5,6,6);
    // std::cout<<"Shape: ["<<tensor_.num()<<", "<<tensor_.channels()<<", "<<tensor_.height()<<", "<<tensor_.width()<<"]"<<std::endl;
    // auto pTensor_ = tensor_.cpu_data();
    // std::cout<<pTensor_<<":data[0]:"<<pTensor_[0]<<std::endl;


    auto x = _Input({1,3,3,3});
    auto y = _SiLU("silu1",{x});
    x = _MatMul("matmul1", {x, y});
    x = _Scale("scale1",{x});
    x = _SoftMax("softmax1",{x}, -1);

    // 输出连接的 EOP
    NetParameter netParam;
    createNetParem(x, netParam);



    // NetParameter netParam;

    // 初始化 netParam 的成员变量
    // netParam.input_name = "input";
    // netParam.output_name = "output";

    // NetOp op1 = {OpType::Silu, {0}, {0},{"input1"}, "silu1"};
    // NetOp op2 = {OpType::Add, {0}, {0}, {"input1", "silu1"}, "add1"};
    // NetOp op3 = {OpType::Matmul, {0}, {0}, {"add1", "input1"}, "matmul1"};

    // netParam.net_ops.push_back(op1);
    // netParam.net_ops.push_back(op2);
    // netParam.net_ops.push_back(op3);


    BackendConfig bn;

    Net net(netParam, bn);
    net.Convert();
    // net.Run();

    Executor ex(&net);
    ex.Execute();
    return 0;
}
