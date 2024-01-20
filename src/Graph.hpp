//
// Created by Rongjie Yi.
//

#ifndef MLLM_GRAPH_H
#define MLLM_GRAPH_H
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "express/ExpressBase.hpp"
#include <unordered_map>
using std::unordered_map;

namespace mllm {

class Graph {
public:
    /**
     * \brief Graph
     * \param param NetParameter contains the structure of this graph
     * \param bn Backend like CPU/QNN etc
     * \param external_tensors external tensors from other graph and inter graphs.
     * \param threadCount number of Threads
     */
    explicit Graph(const NetParameter &param, Backend *bn, unordered_map<string, shared_ptr<Tensor>> &external_tensors, int threadCount);
    virtual ~Graph() = default;

    /**
     * \brief set the output tensors' shape of Ops in this graph.
     */
    void reshape();

    /**
     * \brief alloc the memory of output tensors of Ops in this graph.
     */
    void setUpTensors();

    /**
     * \brief load the weights/bias of Ops in this graph.
     * \param loader A Paramloader
     */
    void setUpOps(ParamLoader &loader);

    /**
     * \brief forward propagation
     * \param autofree Whether to release the memory of weights. Set to false
     * \return The last output tensor
     */
    const vector<shared_ptr<Tensor>> &forward(bool autofree = false);

    /**
     * \brief free the memory of Ops' weights in this graph.
     */
    void freeOps();
    /**
     * \brief free the memory of output tensors of Ops in this graph.
     */
    void freeTensors();
    /**
     * \brief free output tensors & Ops' weights
     */
    void free();

    /**
     * \brief backward propagation [Not Used]
     */
    void backward();

    /**
     * \brief reflash 'ops_input_tensors_'.
     * \param external_tensors external tensors from other graph and inter graphs.
     */
    void reflashInput(unordered_map<string, shared_ptr<Tensor>> &external_tensors);

protected:
    Backend *backend_;
    string name_;

    vector<string> layer_names_;

    // tensor indices for the input and the output of the net
    vector<int> input_tensor_indices_;
    vector<int> output_tensor_indices_;
    vector<Tensor *> input_tensors_;
    vector<Tensor *> output_tensors_;

    unordered_map<string, vector<shared_ptr<Tensor>>> ops_input_tensors_;  // opname: op's output Tensors
    unordered_map<string, vector<shared_ptr<Tensor>>> ops_output_tensors_; // opname: op's output Tensors
    unordered_map<string, shared_ptr<Tensor>> tensors_;                    // opname: Tensors
    unordered_map<string, shared_ptr<Op>> ops_;                            // opname: op

    vector<string> op_names_;

    vector<string> ops_connect_input_;
};

} // namespace mllm

#endif // MLLM_GRAPH_H
