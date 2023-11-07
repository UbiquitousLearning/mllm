#ifndef MLLM_EXECUTOR_H
#define MLLM_EXECUTOR_H
#include "Net.hpp"

namespace mllm {
class Executor {
public:
    Executor() = delete;
    Executor(Net *net) :
        net_(net), data_loader_(nullptr) {
        // nothing to do
        init();
    }
    Executor(Net *net, ParamLoader *data_loader) :
        net_(net), data_loader_(data_loader) {
        // nothing to do
        init();
    }
    ~Executor() = default;

    /**
     * @brief 初始化
     * 使用几个线程，什么策略？
     */
    void init();

    /*
    void graphShapeInit(shared_ptr<Graph> subGraph, unordered_map<string, shared_ptr<Tensor>> &external_tensors) {
        // auto subGraph = net_.subGraph()[graph_name];
        subGraph->reshape(external_tensors);
    }

    void graphSetUp(shared_ptr<Graph> subGraph) {
        // auto subGraph = net_.subGraph()[graph_name];
        subGraph->setUp();
    }

    void graphReshapeOutputs(shared_ptr<Graph> subGraph, unordered_map<string, shared_ptr<Tensor>> &external_tensors) {
        // auto subGraph = net_.subGraph()[graph_name];
        subGraph->reshapeOutputs(external_tensors);
    }

    const vector<shared_ptr<Tensor>> &graphForward(shared_ptr<Graph> subGraph) {
        // auto subGraph = net_.subGraph()[graph_name];
        return subGraph->forward();
    }
    */

    bool checkReshape(bool &init, bool &reshape, vector<int> input_size) {
        if (input_size_.empty()) {
            input_size_ = input_size;
            init = true;
        } else if (input_size.empty()) {
            reshape = false;
        } else if (input_size[0] == input_size_[0] && input_size[1] == input_size_[1] && input_size[2] == input_size_[2] && input_size[3] == input_size_[3]) {
            reshape = false;
        } else {
            input_size_ = input_size;
            reshape = true;
        }
        return init || reshape;
    }

    void execute(vector<int> input_size = {});

    void execute(shared_ptr<Tensor> input_tensor);

    vector<shared_ptr<Tensor>> &result() {
        return result_;
    }

private:
    Net *net_;
    vector<int> input_size_;
    // map<string, map<string,vector<int>>> graph_input_shapes_;
    vector<shared_ptr<Tensor>> result_;
    ParamLoader *data_loader_;

};

} // namespace mllm

#endif // MLLM_EXECUTOR_H
