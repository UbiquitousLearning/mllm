#ifndef MLLM_EXECUTOR_H
#define MLLM_EXECUTOR_H
#include "Net.hpp"
#include<numeric>

namespace mllm {
class Executor {
public:
//    Executor() = delete;
    Executor():
        data_loader_(nullptr) {
        // nothing to do
        init();
    }
    Executor(ParamLoader *data_loader) :
        data_loader_(data_loader) {
        // nothing to do
        // init();
    }
    ~Executor() = default;

    /**
     * @brief 初始化
     * 使用几个线程，什么策略？
     */
    void init();
    void setup(Net *net, vector<shared_ptr<Tensor>> input_tensors);
    void run(Net *net, vector<shared_ptr<Tensor>> input_tensors);

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
    bool checkSame(vector<shared_ptr<Tensor>> input_tensor){
        if(input_tensor.size() != input_sizes_.size()){
            return false;
        }
        bool same = true;
        for (int i = 0; i < input_tensor.size(); ++i) {
            if (input_tensor[i]->shape() != input_sizes_[i]) {
                same = false;
                break;
            }
        }
        return same;
    }

    bool checkReshape(bool &init, bool &reshape, vector<shared_ptr<Tensor>> input_tensor) {
        if (input_sizes_.empty()) {
            for (auto &t : input_tensor) {
                input_sizes_.push_back(t->shape());
            }
            init = true;
        } else if (checkSame(input_tensor)) {
            reshape = false;
        } else {
            input_sizes_.clear();
            for (auto &t : input_tensor) {
                input_sizes_.push_back(t->shape());
            }
            reshape = true;
        }
        return init || reshape;
    }

    void execute(vector<int> input_size = {});

    void execute(Net *net, vector<shared_ptr<Tensor>> input_tensor);

    vector<shared_ptr<Tensor>> &result() {
        return result_;
    }

    void perf() const{
        std::cout << "load time: " << load_time_ << " ms" << std::endl;
        //average of run_time_
        double sum_time = std::accumulate(std::begin(run_time_), std::end(run_time_), 0.0);
        double mean_time =  sum_time / run_time_.size();
        std::cout << "token time: " << mean_time<< " ms"<<std::endl;
        std::cout << "inference speed: " << 1000 /mean_time << " tokens/s" << std::endl;
    }

private:
    vector<vector<int>> input_sizes_;
    // map<string, map<string,vector<int>>> graph_input_shapes_;
    vector<shared_ptr<Tensor>> result_;
    ParamLoader *data_loader_;


    double load_time_ = 0;
    vector<double> run_time_;

};

} // namespace mllm

#endif // MLLM_EXECUTOR_H
