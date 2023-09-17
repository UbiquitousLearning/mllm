
#ifndef MLLM_NET_H
#define MLLM_NET_H


#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Graph.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <vector>
namespace mllm
{
    class Net {
    public:
        explicit Net(const NetParameter& param,BackendConfig config);
        virtual ~Net() = default;

        void Convert();

        unordered_map<string, shared_ptr<Graph>>& subGraphFP() {
            return subgraphs_fp_;
        }

    private:
        NetParameter net_param_;
        unordered_map<string, shared_ptr<Graph>> subgraphs_fp_;
        // unordered_map<string, shared_ptr<Graph>> subgraphs_int8_;
        BackendConfig config_;
        vector<Tensor*> tensors_;
        vector<Op*> ops_;        
        unordered_map<BackendType, Backend*> backends_;
        ParamLoader *data_loader_;
    };
    
} // namespace mllm


#endif //MLLM_NET_H