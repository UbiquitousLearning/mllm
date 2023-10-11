#ifndef MLLM_EXECUTOR_H
#define MLLM_EXECUTOR_H
#include "Net.hpp"

namespace mllm {
class Executor {
public:
    Executor() = delete;
    Executor(Net *net) :
        net_(net) {
        // nothing to do
    }
    ~Executor() = default;

    /**
     * @brief 初始化
     * 使用几个线程，什么策略？
     */
    void init();

    void graphShapeInit(shared_ptr<Graph> subGraph, unordered_map<string, shared_ptr<Tensor>> &external_tensors) {
        // auto subGraph = net_.subGraph()[graph_name];
        subGraph->shapeInit(external_tensors);
    }

    void graphSetUp(shared_ptr<Graph> subGraph) {
        // auto subGraph = net_.subGraph()[graph_name];
        subGraph->setUp();
    }

    void graphReshapeOutputs(shared_ptr<Graph> subGraph, unordered_map<string, shared_ptr<Tensor>> &external_tensors) {
        // auto subGraph = net_.subGraph()[graph_name];
        subGraph->reshapeOutputs(external_tensors);
    }

    /**
     * @brief 前行传播
     */
    const vector<shared_ptr<Tensor>> &graphForward(shared_ptr<Graph> subGraph) {
        // auto subGraph = net_.subGraph()[graph_name];
        return subGraph->forward();
        // TODO: 在此处插入 return 语句
    }

    void execute(vector<int> input_size ={});

private:
    Net *net_;
    vector<int> input_size_;
};

} // namespace mllm

#endif // MLLM_EXECUTOR_H
