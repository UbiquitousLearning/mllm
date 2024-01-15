//
// Created by yirongjie.
//

#ifndef MLLM_GRAPH_H
#define MLLM_GRAPH_H
#include "express/Express.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"

// using std::unordered_map;
#include <unordered_map>
using std::unordered_map;
// #include "layer.h"
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

    //void reshapeOutputs();

    //void setUp(unordered_map<string, shared_ptr<Tensor>> &external_tensors, bool init, bool reshape, bool graph0);


    /**
     * @brief 前行传播
     */
    // const  vector<shared_ptr<Tensor>>& forward();
    const vector<shared_ptr<Tensor>> &forward(bool autofree = false);
    // set input blobs then use forward() instead.
    //const vector<shared_ptr<Tensor>> &forward(const vector<shared_ptr<Tensor>> &inTensors);

    void freeOps();
    void freeTensors();
    void free();

//    const vector<shared_ptr<Tensor>> &inputTensors(){
//        return ops_input_tensors_[param_.net_ops[0]->name];
//    }
//    const vector<shared_ptr<Tensor>> &outputTensors(){
//        return ops_output_tensors_[param_.net_ops[param_.net_ops.size() - 1]->name];
//    }

    /**
     * @brief 反向传播
     */
    void backward();

//    NetParameter &param() {
//        return param_;
//    }

    void reflashInput(unordered_map<string, shared_ptr<Tensor>> &external_tensors);

protected:
//    NetParameter param_;
    Backend *backend_;
    // The network name
    string name_;
    // The phase: TRAIN or TEST
    // Phase phase_;

    // Individual layers in the net
    // vector<shared_ptr<Layer > > layers_;
    vector<string> layer_names_;

    // tensor indices for the input and the output of the net
    vector<int> input_tensor_indices_;
    vector<int> output_tensor_indices_;
    vector<Tensor *> input_tensors_;
    vector<Tensor *> output_tensors_;

    // vector <string> op_names_;
    // vector<vector<string>> op_in_names_;
    unordered_map<string, vector<shared_ptr<Tensor>>> ops_input_tensors_;  // opname: op's output Tensors
    unordered_map<string, vector<shared_ptr<Tensor>>> ops_output_tensors_; // opname: op's output Tensors
    unordered_map<string, shared_ptr<Tensor>> tensors_;                    // opname: Tensors
    unordered_map<string, shared_ptr<Op>> ops_;                            // opname: op
    //    unordered_map<string, shared_ptr<Tensor>> external_tensors_;

    vector<string> op_names_;

    vector<string> ops_connect_input_; // opname: op's input Tensors

};

} // namespace mllm

#endif // MLLM_GRAPH_H
