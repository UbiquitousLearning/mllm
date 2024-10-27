#ifndef MLLM_NET_H
#define MLLM_NET_H

#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Graph.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
namespace mllm {
class Net {
public:
    explicit Net(BackendConfig config);
    virtual ~Net() = default;

    virtual void convert(vector<NetParameter> &param, BackendType backend_type = BackendType::MLLM_CPU, int threadCount = 4);

    unordered_map<string, shared_ptr<Graph>> &subGraph() {
        return subGraphs_;
    }
    unordered_map<string, shared_ptr<Tensor>> &tensors() {
        return tensors_;
    }

    unordered_map<BackendType, shared_ptr<Backend>> &backends() {
        return backends_;
    }
    vector<vector<string>> &tensorNames() {
        return tensor_names_;
    }
    void freeTensors(int graph_idx);
    vector<string> inputNames() const {
        return input_names_;
    }
    map<string, int> inGmap() const {
        return inputname_graphidx_;
    }

protected:
    unordered_map<string, shared_ptr<Graph>> subGraphs_;
    unordered_map<string, shared_ptr<Tensor>> tensors_;
    vector<vector<string>> tensor_names_;
    vector<NetOp *> ops_;
    unordered_map<BackendType, shared_ptr<Backend>> backends_;
    vector<string> input_names_;
    map<string, int> inputname_graphidx_;
};

} // namespace mllm

#endif // MLLM_NET_H