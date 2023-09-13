#include "Executor.hpp"
namespace mllm
{
    void Executor::Init()
    {
    }

    void Executor::GraphSetup(shared_ptr<Graph<float>> subGraph)
    {
        // auto subGraph = net_.subGraphFP()[graph_name];
        subGraph->Setup();
    }

    const vector<shared_ptr<Tensor<float>>> &Executor::GraphForward(shared_ptr<Graph<float>> subGraph)
    {
        // auto subGraph = net_.subGraphFP()[graph_name];
        return subGraph->Forward();
        // TODO: 在此处插入 return 语句
    }

    void Executor::Execute()
    {
        for(auto kv: net_->subGraphFP()){
            std::cout<<kv.first<<std::endl;
            GraphSetup(kv.second);
            auto result = GraphForward(kv.second);
        }
        
    }

} // namespace mllm
