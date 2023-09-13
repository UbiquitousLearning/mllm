#include "Net.hpp"
#include "backends/cpu/CPUBackend.hpp"
namespace mllm
{
    Net::Net(const NetParameter &param)
    {
        net_param_ = param;
    }

    void Net::Convert(shared_ptr<MemoryManager> p_mm)
    {
        // TODO
        // auto bn = new Backend();
        
        // shared_ptr<MemoryManager> p_mm(new MemoryManager());
        auto bn = new CPUBackend(p_mm);	//TODO
        backends_["cpu"] = bn;
        // TODO
        auto sub_param_ = net_param_;
        shared_ptr<Graph<float>> subg_fp1;
        subg_fp1.reset(new Graph<float>(sub_param_, backends_["cpu"]));
        subgraphs_fp_["fp1"] = subg_fp1;
    }

    // const vector<shared_ptr<Tensor<float>>> &Net::Run()
    // {
    //     // TODO
    //     for(auto& kv:subgraphs_fp_){
    //         kv.second->Setup();
    //         kv.second->Forward();
    //     }
    //     // TODO: 在此处插入 return 语句 
    //     //return;
    // }

} // namespace mllm
