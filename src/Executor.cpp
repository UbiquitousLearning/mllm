#include <csignal>
#include "Timing.hpp"
#include "Executor.hpp"

namespace mllm {
void Executor::setup(Net *net) {
    mllm_time_init();

    uint64_t time_start = mllm_time_us();
    uint64_t time_end;

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];

        g->setUpOps(*data_loader_);
    }
    time_end = mllm_time_us();
    if (load_time_ == 0) {
        load_time_ = (time_end - time_start) / 1000.0F;
        std::cout << "Load model: " << load_time_ / 1000.0F << " s" << std::endl;
    }
}

void Executor::run(Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    bool init = false;
    bool reshape = false;

    checkReshape(init, reshape, input_tensors);

    // set Input tensor
    vector<int> flashGid = {};
    for (int tid = 0; tid < net->inputNames().size(); ++tid) {
        auto input_name = net->inputNames()[tid];
        auto input_tensor = input_tensors[tid];
        input_tensor->setName(input_name);
        net->tensors()[input_name] = input_tensor;
        if (std::find(flashGid.begin(), flashGid.end(), net->inGmap()[input_name]) == flashGid.end()) {
            flashGid.push_back(net->inGmap()[input_name]);
        }
    }
    for (auto Gid : flashGid) {
        net->subGraph()["G" + std::to_string(Gid)]->reflashInput(net->tensors());
    }

    auto ex_time_start = mllm_time_us();

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        std::cout << "=======forwarding graph " << i << std::endl;
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];

        g->reshape();
        g->setUpTensors();

        result_ = g->forward();

        // free
        if (false) {
            if (i < (int)net->subGraph().size() - 1) {
                g->freeTensors();
            }
            net->freeTensors(i);
        }
    }
    std::cout << "result size" << result_.size() << std::endl;
    auto ex_time_end = mllm_time_us();
    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }
    auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
    run_time_.push_back(token_run_time);
}

// #define DYNAMIC
void Executor::execute(Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    bool init = false;
    bool reshape = false;
    // TODO: when reshape begin
    checkReshape(init, reshape, input_tensors);
    // set Input tensor

    uint64_t time_start = mllm_time_us();
    uint64_t time_end;

    // Init inputs
    vector<int> flashGid = {};
    for (int tid = 0; tid < net->inputNames().size(); ++tid) {
        auto input_name = net->inputNames()[tid];
        auto input_tensor = input_tensors[tid];
        input_tensor->setName(input_name);
        net->tensors()[input_name] = input_tensor;
        if (std::find(flashGid.begin(), flashGid.end(), net->inGmap()[input_name]) == flashGid.end()) {
            flashGid.push_back(net->inGmap()[input_name]);
        }
    }
    for (auto Gid : flashGid) {
        net->subGraph()["G" + std::to_string(Gid)]->reflashInput(net->tensors());
    }

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];
        if (init || reshape) {
            g->reshape();
        }
        // load params
        if (!paramloaded) {
            g->setUpOps(*data_loader_);
        }
#ifndef DYNAMIC
    }
    paramloaded = true;
    time_end = mllm_time_us();
    if (load_time_ == 0) {
        load_time_ = (time_end - time_start) / 1000.0F;
    }

    auto ex_time_start = mllm_time_us();
    float exe_time = 0;

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];
#endif

        g->reshape();
        g->setUpTensors();

        result_ = g->forward();

        // free
        if (freeGraph) {
#ifdef DYNAMIC
            g->freeOps();
            paramloaded = false;
#endif
            if (i < (int)net->subGraph().size() - 1) {
                g->freeTensors();
            }
            net->freeTensors(i);
        }
    }
    auto ex_time_end = mllm_time_us();
    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }
}

} // namespace mllm
