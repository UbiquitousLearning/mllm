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

    void convert();
    // TODO: remove
    // convert all subgraph to specified backend, just for develop
    void convert(BackendType backend_type);

    void reshapeInput();
    void reshapeInput(vector<int> shape);
    void setInput();

    unordered_map<string, shared_ptr<Graph>> &subGraph() {
        return subGraphs_;
    }
    unordered_map<string, shared_ptr<Tensor>> &tensors() {
        return tensors_;
    }
    vector<NetParameter> &netParam() {
        return net_param_;
    }
    unordered_map<BackendType, Backend *> &backends() {
        return backends_;
    }
    vector<vector<string>> &tensorNames() {
        return tensor_names_;
    }
    void freeTensors(int graph_idx);

private:
    vector<NetParameter> net_param_;
    unordered_map<string, shared_ptr<Graph>> subGraphs_;
    BackendConfig config_;
    // vector<Tensor *> tensors_;
    unordered_map<string, shared_ptr<Tensor>> tensors_;
    vector<vector<string>> tensor_names_;
    vector<NetOp *> ops_;
    unordered_map<BackendType, Backend *> backends_;
    //    ParamLoader *data_loader_;
};

} // namespace mllm

#endif // MLLM_NET_H