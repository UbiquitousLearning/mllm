#include "Net.hpp"
namespace mllm
{
    Net::Net(const NetParameter &param)
    {
        net_param_ = param;
    }

    void Net::Convert()
    {
        // TODO
        auto sub_param_ = net_param_;
        shared_ptr<Graph<float>> subg_fp1;
        subg_fp1.reset(new Graph<float>(sub_param_));
        subgraphs_fp_["fp1"] = subg_fp1;
    }

    const vector<shared_ptr<Tensor<float>>> &Net::Run()
    {
        // TODO
        for(auto& kv:subgraphs_fp_){
            kv.second->Setup();
            kv.second->Forward();
        }
        // TODO: 在此处插入 return 语句 
        //return;
    }

} // namespace mllm
