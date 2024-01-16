//
// Created by Rongjie Yi.
//

#ifndef MLLM_GRAPH_H
#define MLLM_GRAPH_H
#include "express/Express.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"


#include <unordered_map>
using std::unordered_map;

namespace mllm {

class Graph {
public:
    explicit Graph(const NetParameter &param, Backend *bn, unordered_map<string, shared_ptr<Tensor>> &external_tensors, int threadCount);
    virtual ~Graph() = default;

    /**
     * @brief 初始化
     */
    void reshape();

    void setUpTensors();

    void setUpOps(ParamLoader &loader);

    /**
     * @brief forward
     */
    const vector<shared_ptr<Tensor>> &forward(bool autofree = false);

    void freeOps();
    void freeTensors();
    void free();

    /**
     * @brief backward
     */
    void backward();

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

    vector<string> ops_connect_input_; // opname: op's input Tensors
};

} // namespace mllm

#endif // MLLM_GRAPH_H
