#ifndef MLLM_QNNNET_H
#define MLLM_QNNNET_H

#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Graph.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "Net.hpp"
#include "ExpressBase.hpp"

namespace mllm {
class QNNNet : public Net {
public:
    explicit QNNNet(BackendConfig config, Context *ctx);
    virtual ~QNNNet() = default;

    virtual void convert(vector<NetParameter> &param, BackendType backend_type = BackendType::MLLM_QNN, int threadCount=4) override;

    /*
     * Net functions
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
    vector<string> inputNames() const{
        return input_names_;
    }
    map<string, int> inGmap() const{
        return inputname_graphidx_;
    }

    */

private:

    void build_new_graph(NetOp *op);

    /*
     * Net variables
    unordered_map<string, shared_ptr<Graph>> subGraphs_;
    unordered_map<string, shared_ptr<Tensor>> tensors_;
    vector<vector<string>> tensor_names_;
    vector<NetOp *> ops_;
    unordered_map<BackendType, shared_ptr<Backend>> backends_;
    vector<string> input_names_ ;
    map<string, int> inputname_graphidx_;
    */

    Context *ctx_;
    std::string Quantizationtype = "Smoothquant";
    
    "k-quant"

};

} // namespace mllm

#endif // MLLM_NET_H