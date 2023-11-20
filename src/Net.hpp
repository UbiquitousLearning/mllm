#ifndef MLLM_NET_H
#define MLLM_NET_H

#include "NetParameter.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Graph.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <vector>
namespace mllm {
class Net {
public:
    explicit Net(const vector<NetParameter> &param, BackendConfig config);
    virtual ~Net() = default;

    // convert all subgraph to specified backend, just for develop
    void convert(vector<NetParameter> &param, BackendType backend_type = BackendType::MLLM_CPU);

    unordered_map<string, shared_ptr<Graph>> &subGraph() {
        return subGraphs_;
    }
    unordered_map<string, shared_ptr<Tensor>> &tensors() {
        return tensors_;
    }

    unordered_map<BackendType,  shared_ptr<Backend>> &backends() {
        return backends_;
    }
    vector<vector<string>> &tensorNames() {
        return tensor_names_;
    }
    void freeTensors(int graph_idx);
    string inputName() {
        return input_name_;
    }

private:
    unordered_map<string, shared_ptr<Graph>> subGraphs_;
    BackendConfig config_;
    unordered_map<string, shared_ptr<Tensor>> tensors_;
    vector<vector<string>> tensor_names_;
    vector<NetOp *> ops_;
    unordered_map<BackendType, shared_ptr<Backend>> backends_;
    string input_name_ ;

};

} // namespace mllm

#endif // MLLM_NET_H