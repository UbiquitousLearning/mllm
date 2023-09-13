
#ifndef MLLM_NET_H
#define MLLM_NET_H


#include "Graph.hpp"
namespace mllm
{
    class Net {
    public:
        explicit Net(const NetParameter& param);
        virtual ~Net() = default;

        void Convert(shared_ptr<MemoryManager> p_mm);

        // /**
        //  * @brief 执行，用户可重构
        //  */
        // const vector<shared_ptr<Tensor>>& Run();


        unordered_map<string, shared_ptr<Graph>>& subGraphFP() {
            return subgraphs_fp_;
        }

    private:
        NetParameter net_param_;
        unordered_map<string, shared_ptr<Graph>> subgraphs_fp_;
        // unordered_map<string, shared_ptr<Graph>> subgraphs_int8_;

        
        unordered_map<string, Backend*> backends_;
    };
    
} // namespace mllm


#endif //MLLM_NET_H