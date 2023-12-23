#include <csignal>
#include "Executor.hpp"
namespace mllm {
void Executor::setup(Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    mllm_time_init();
    bool init = true;
    // set Input tensor

    uint64_t time_start = mllm_time_us();
    uint64_t time_end;
    uint64_t t_start;
    uint64_t t_end;
    //Init inputs
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
    for (auto Gid :flashGid) {
        net->subGraph()["G" + std::to_string(Gid)]->reflashInput(net->tensors());
    }

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];
#ifdef DEBUG
            std::cout << "[" << name << "]==== reshape";
            t_start = mllm_time_us();
#endif
            g->reshape();
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
#ifdef DEBUG
            std::cout << "[" << name << "]==== load";
            t_start = mllm_time_us();
#endif
            g->setUpOps(*data_loader_);
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << "    ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
    }
    time_end = mllm_time_us();
    if (load_time_ == 0) {
        load_time_ = (time_end - time_start) / 1000.0F;
        std::cout << "Load model: " << load_time_/1000.0F << " s" << std::endl;
    }
}

void Executor::run(Net *net, vector<shared_ptr<Tensor>> input_tensors) {
//    auto input_size = input_tensor->shape();
    bool init = false;
    bool reshape = false;
    // TODO: when reshape begin
    checkReshape(init, reshape, input_tensors);
    // set Input tensor

    uint64_t t_start;
    uint64_t t_end;
    uint64_t time_start = mllm_time_us();
    uint64_t time_end;
    //Init inputs
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
    for (auto Gid :flashGid) {
        net->subGraph()["G" + std::to_string(Gid)]->reflashInput(net->tensors());
    }

    // for (int i = 0; i < (int)net->subGraph().size(); ++i) {
    //     string name = "G" + std::to_string(i);
    //     auto &g = net->subGraph()[name];
    //     if (reshape) {
    //         g->reshape();
    //     }
    // }
    auto ex_time_start = mllm_time_us();
    float exe_time = 0;

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];
#ifdef DEBUG
        std::cout << "[" << name << "]==== setup";
        t_start = mllm_time_us();
#endif
        g->reshape();
        g->setUpTensors();
#ifdef DEBUG
        t_end = mllm_time_us();
        std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
// exe
#ifdef DEBUG
        std::cout << "[" << name << "]==== execute";
        t_start = mllm_time_us();
#endif
        result_ = g->forward();
#ifdef DEBUG
        t_end = mllm_time_us();
        exe_time += (t_end - t_start) / 1000.0F;
        std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        // free
        if (false) {
#ifdef DEBUG
            std::cout << "[" << name << "]==== free";
            t_start = mllm_time_us();
#endif
#ifdef DYNAMIC
            g->freeOps();
            paramloaded = false;
#endif
            if (i < (int)net->subGraph().size() - 1) {
                g->freeTensors();
            }
            net->freeTensors(i);
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << "    ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        }
        // std::cout <<"["<< name << "]==== end      === "<< result_[0]->name() << "'s shape:  [" << result_[0]->batch() << "," << result_[0]->head() << "," << result_[0]->sequence() << "," << result_[0]->dimension() << "]" << std::endl;
    }
    auto ex_time_end = mllm_time_us();
    // std::cout <<"                "<< (ex_time_end - ex_time_start) / 1000.0F << " ms" << std::endl;
    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
#ifdef DEBUG
        std::cout << "exec ====  "<<exe_time<<" ms ====="<< (ex_time_end - ex_time_start) / 1000.0F << " ms" << std::endl;
#endif
    }
}

// #define DEBUG
// #define DYNAMIC
bool paramloaded = false;
bool freeGraph = false;
void Executor::execute(Net *net, vector<shared_ptr<Tensor>> input_tensors) {
//    auto input_size = input_tensor->shape();
    bool init = false;
    bool reshape = false;
    // TODO: when reshape begin
    checkReshape(init, reshape, input_tensors);
    // set Input tensor

    uint64_t t_start;
    uint64_t t_end;
    uint64_t time_start = mllm_time_us();
    uint64_t time_end;

//    input_tensor->setName(net->inputName());
//    net->tensors()[net->inputName()] = input_tensor;
//    net->subGraph()["G0"]->reflashInput(net->tensors(), net->inputName());
    //Init inputs
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
    for (auto Gid :flashGid) {
        net->subGraph()["G" + std::to_string(Gid)]->reflashInput(net->tensors());
    }

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];
        if (init || reshape) {
#ifdef DEBUG
            std::cout << "[" << name << "]==== reshape";
            t_start = mllm_time_us();
#endif
            g->reshape();
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        }
        // load params
        if (!paramloaded) {
#ifdef DEBUG
            std::cout << "[" << name << "]==== load";
            t_start = mllm_time_us();
#endif
            g->setUpOps(*data_loader_);
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << "    ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        }
#ifndef DYNAMIC
    }
    paramloaded = true;
    time_end = mllm_time_us();
    if (load_time_ == 0) {
        load_time_ = (time_end - time_start) / 1000.0F;
    }
#ifdef DEBUG
    std::cout << "reshape&load ====  " << (time_end - time_start) / 1000.0F << " ms" << std::endl;
#endif
    auto ex_time_start = mllm_time_us();
    float exe_time = 0;

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];
#endif
        // if (init || reshape)
//        {
#ifdef DEBUG
        std::cout << "[" << name << "]==== setup";
        t_start = mllm_time_us();
#endif
        g->reshape();
        g->setUpTensors();
#ifdef DEBUG
        t_end = mllm_time_us();
        std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
//        }
// exe
#ifdef DEBUG
        std::cout << "[" << name << "]==== execute";
        t_start = mllm_time_us();
#endif
        result_ = g->forward();
#ifdef DEBUG
        t_end = mllm_time_us();
        exe_time += (t_end - t_start) / 1000.0F;
        std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        // free
        if (freeGraph) {
#ifdef DEBUG
            std::cout << "[" << name << "]==== free";
            t_start = mllm_time_us();
#endif
#ifdef DYNAMIC
            g->freeOps();
            paramloaded = false;
#endif
            if (i < (int)net->subGraph().size() - 1) {
                g->freeTensors();
            }
            net->freeTensors(i);
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << "    ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        }
        // std::cout <<"["<< name << "]==== end      === "<< result_[0]->name() << "'s shape:  [" << result_[0]->batch() << "," << result_[0]->head() << "," << result_[0]->sequence() << "," << result_[0]->dimension() << "]" << std::endl;
    }
    auto ex_time_end = mllm_time_us();
    // std::cout <<"                "<< (ex_time_end - ex_time_start) / 1000.0F << " ms" << std::endl;
    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
#ifdef DEBUG
        std::cout << "exec ====  "<<exe_time<<" ms ====="<< (ex_time_end - ex_time_start) / 1000.0F << " ms" << std::endl;
#endif
    }
}

} // namespace mllm
