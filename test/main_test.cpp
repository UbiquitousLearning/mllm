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


    NetParameter param;
    param.bntype = mllm_CPU;
    vector < string > name_ = {"mm1", "mm2"};
    param.op_names_ = name_;
    vector<vector<string>> io_name_ = { {"input", "input"}, {"input", "mm1"}};
    param.op_in_names_ = io_name_;

    shared_ptr<MemoryManager> p_mm(new MemoryManager());

    Net net(param);
    net.Convert(p_mm);
    // net.Run();

    Executor ex(&net);
    ex.Execute();
    // ex.GraphSetup("fp1");
    // auto result = ex.GraphForward("fp1");
    return 0;
}