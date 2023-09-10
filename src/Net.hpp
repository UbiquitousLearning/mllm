
#ifndef MLLM_NET_H
#define MLLM_NET_H


#include "Graph.hpp"
namespace mllm
{
    class Net {
    public:
        explicit Net(const NetParameter& param);
        virtual ~Net() = default;

        void Convert();

        /**
         * @brief 执行，用户可重构
         */
        const vector<shared_ptr<Tensor<float>>>& Run();
    private:
        NetParameter net_param_;
        unordered_map<string, shared_ptr<Graph<float>>> subgraphs_fp_;
        unordered_map<string, shared_ptr<Graph<int8_t>>> subgraphs_int8_;
    };
    
} // namespace mllm


#endif //MLLM_NET_H