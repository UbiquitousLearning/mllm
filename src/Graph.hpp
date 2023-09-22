//
// Created by yirongjie.
//

#ifndef MLLM_GRAPH_H
#define MLLM_GRAPH_H
#include "NetParameter.hpp"
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
    explicit Graph(const NetParameter &param, Backend *bn, unordered_map<string, shared_ptr<Tensor>> &external_tensors);
    virtual ~Graph() = default;

    /**
     * @brief 初始化
     */
    void init(unordered_map<string, shared_ptr<Tensor>> &external_tensors);

    /**
     * @brief 初始化
     */
    void setUp();

    void load(ParamLoader &loader);

    /**
     * @brief 前行传播
     */
    // const  vector<shared_ptr<Tensor>>& forward();
    const vector<shared_ptr<Tensor>> &forward();
    // set input blobs then use forward() instead.
    const vector<shared_ptr<Tensor>> &forward(const vector<shared_ptr<Tensor>> &inTensors);

    /**
     * @brief 反向传播
     */
    void backward();

protected:
    NetParameter param_;
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
};

} // namespace mllm

#endif // MLLM_GRAPH_H
