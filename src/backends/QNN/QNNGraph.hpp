#ifndef MLLM_QNNGRAPH_H
#define MLLM_QNNGRAPH_H
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "express/ExpressBase.hpp"
#include <unordered_map>
#include <Graph.hpp>
#include <thread>

using std::unordered_map;

namespace mllm {

class QNNGraph : public Graph {
public:
    /**
     * \brief Graph
     * \param param NetParameter contains the structure of this graph
     * \param bn Backend like CPU/QNN etc
     * \param external_tensors external tensors from other graph and inter graphs.
     * \param threadCount number of Threads
     */
    explicit QNNGraph(const NetParameter &param, Backend *bn, unordered_map<string, shared_ptr<Tensor>> &external_tensors, int threadCount);
    virtual ~QNNGraph() = default;

    /**
     * \brief forward propagation
     * \param autofree Whether to release the memory of weights. Set to false
     * \return The last output tensor
     */
    virtual const vector<shared_ptr<Tensor>> &forward(bool autofree = false) override;

    // TODO: WARNING!!! non virtual forward
    const vector<shared_ptr<Tensor>> &forward(std::string graphName);

    void setUpTensors(std::string graphName);
    void free(std::string graphName);
    void allFree();

    void QNNThreadExecute();

private:
    uint autoregressive_seq_pos_ = 0;

};

} // namespace mllm

#endif // MLLM_GRAPH_H
