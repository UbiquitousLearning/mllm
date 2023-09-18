#include <iostream>
// #include "Graph.hpp"
// #include "Tensor.hpp"
// #include "backends/cpu/CPUMatmul.hpp"
#include "Net.hpp"
#include "Executor.hpp"

using namespace mllm;

int main()
{
    // Tensor tensor_(1,3,5,5);
    // std::cout << "Init Tensor" << std::endl;
    // tensor_.Reshape(1,5,6,6);
    // std::cout<<"Shape: ["<<tensor_.num()<<", "<<tensor_.channels()<<", "<<tensor_.height()<<", "<<tensor_.width()<<"]"<<std::endl;
    // auto pTensor_ = tensor_.cpu_data();
    // std::cout<<pTensor_<<":data[0]:"<<pTensor_[0]<<std::endl;


    // CPUMatmul mm_op(mllm_CPU,true,true,true,true);


    // NetParameter param;
    // vector < string > name_ = {"mm1", "mm2"};
    // param.op_names_ = name_;
    // vector<vector<string>> io_name_ = { {"input", "input"}, {"input", "mm1"}};
    // param.op_in_names_ = io_name_;

    NetParameter netParam;

    // 初始化 netParam 的成员变量
    netParam.input_name = "input";
    netParam.output_name = "output";

    NetOp op1 = {OpType::Silu, {0}, {0},{"input1"}, "silu1"};
    NetOp op2 = {OpType::Add, {0}, {0}, {"input1", "silu1"}, "add1"};
    NetOp op3 = {OpType::Matmul, {0}, {0}, {"add1", "input1"}, "matmul1"};

    netParam.net_ops.push_back(op1);
    netParam.net_ops.push_back(op2);
    netParam.net_ops.push_back(op3);


    BackendConfig bn;

    // shared_ptr<MemoryManager> p_mm(new MemoryManager());

    Net net(netParam, bn);
    net.Convert();
    // net.Run();

    Executor ex(&net);
    ex.Execute();
    // ex.GraphSetup("fp1");
    // auto result = ex.GraphForward("fp1");
    return 0;
}