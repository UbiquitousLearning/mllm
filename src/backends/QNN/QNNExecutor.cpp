#include <csignal>
#include <fstream>
#include "QNNGraph.hpp"
#include "Timing.hpp"
#include "QNNExecutor.hpp"

namespace mllm {
void QNNExecutor::setup(Net *net) {
    mllm_time_init();

    uint64_t time_start = mllm_time_us();
    uint64_t time_end;

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name;
        if (executionType_ == PROMPT) {
            name = "Prompt_Graph." + std::to_string(i);
        } else if (executionType_ == AUTOREGRESSIVE) {
            name = "Autoregressive_Graph." + std::to_string(i);
        }
            
        auto &g = net->subGraph()[name];

        g->setUpOps(*data_loader_);
    }
    time_end = mllm_time_us();
    if (load_time_ == 0) {
        load_time_ = (time_end - time_start) / 1000.0F;
        std::cout << "Load model: " << load_time_ / 1000.0F << " s" << std::endl;
    }
}

// void QNNExecutor::QNNGraphThreadExecute(int id, Net* net) {

//     string typeName = "Prompt_Graph.";
    
//     std::vector<int> names;
//     std::vector<QNNGraph*> qnn_graphs;
//     for (int i = id; i < (int)net->subGraph().size(); i+=this->threadNum_) {
//         string name = typeName + std::to_string(i);
//         auto &g = net->subGraph()[name];
//         auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        
//         qnn_graphs.push_back(qnn_graph);
//         names.push_back(i);
//     }

//     while(!this->threadVar_[id])
//         usleep(10);

//     auto ex_time_start = mllm_time_us();
//     for (int i = 0; i < names.size(); i++) 
//         result_ = qnn_graphs[i]->forward(typeName + std::to_string(names[i]));
//     auto ex_time_end = mllm_time_us();
//     std::cout << "thread execution time" << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;
// }

void QNNExecutor::run(Net *net, vector<shared_ptr<Tensor>> input_tensors) {
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
    string typeName;
    if (executionType_ == PROMPT) {
        typeName = "Prompt_Graph.";
    } else if (executionType_ == AUTOREGRESSIVE) {
        typeName = "Autoregressive_Graph.";
    }
    for (auto Gid : flashGid) {
        net->subGraph()[typeName + std::to_string(Gid)]->reflashInput(net->tensors());
    }

    std::fstream fs("AR_latency.txt", std::ios::app);
    auto ex_time_start = mllm_time_us();

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];

        // cast graph to QNNGraph
        // TODO: if this implementation is used, the setUpTensors(string) should be merged to Graph
        // the qnn_graph below is where we cast the Graph to QNNGraph
        if(i == 0){
            g->reshape();
            g->setUpTensors();
        } else {
            std::cout << "=======setup qnn graph " << i << std::endl;
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            g->reshape();
            // if ( autoregressive_seq_pos_ % 32 == 31 || autoregressive_seq_pos_ == 0) {
            // g->setUpTensors();
            qnn_graph->setUpTensors(name);
            // }
        }
        if (false) {
            if (i < (int)net->subGraph().size() - 1) {
                g->freeTensors();
            }
            net->freeTensors(i);
        }
    }
    auto ex_time_end = mllm_time_us();

    // std::thread qnnGraphThread_0(&QNNExecutor::QNNGraphThreadExecute, this, 0, net);
    // std::thread qnnGraphThread_1(&QNNExecutor::QNNGraphThreadExecute, this, 1, net);
    // std::thread qnnGraphThread_2(&QNNExecutor::QNNGraphThreadExecute, this, 2, net);
    // std::thread qnnGraphThread_3(&QNNExecutor::QNNGraphThreadExecute, this, 3, net);
    // std::thread qnnGraphThread_4(&QNNExecutor::QNNGraphThreadExecute, this, 4, net);
    // std::thread qnnGraphThread_5(&QNNExecutor::QNNGraphThreadExecute, this, 5, net);
    // std::thread qnnGraphThread_6(&QNNExecutor::QNNGraphThreadExecute, this, 6, net);
    // std::thread qnnGraphThread_7(&QNNExecutor::QNNGraphThreadExecute, this, 7, net);

    fs << "setup all graph" << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;

    ex_time_start = mllm_time_us();

    // execute all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        if(i == 0){
            string name = typeName + std::to_string(i);
            auto &g = net->subGraph()[name];
            result_ = g->forward();
        } else {
            string name = typeName + std::to_string(i);
            auto &g = net->subGraph()[name];
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            result_ = qnn_graph->forward(name);
        }
    }

    // threadVar_[0] = true;
    // threadVar_[1] = true;
    // threadVar_[2] = true;
    // threadVar_[3] = true;
    // threadVar_[4] = true;
    // threadVar_[5] = true;
    // threadVar_[6] = true;
    // threadVar_[7] = true;

    // qnnGraphThread_0.join();
    // qnnGraphThread_1.join();
    // qnnGraphThread_2.join();
    // qnnGraphThread_3.join();
    // qnnGraphThread_4.join();
    // qnnGraphThread_5.join();
    // qnnGraphThread_6.join();
    // qnnGraphThread_7.join();



    ex_time_end = mllm_time_us();
    fs << "execute all graph" << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;

    // free all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        if(i == 0) continue;
        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free(name);
    }

    // first graph all free is OK.
    // {
    //     string name = typeName + std::to_string(0);
    //     auto &g = net->subGraph()[name];
    //     auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
    //     qnn_graph->allFree();
    // }
    
    // open file "AR_latency.txt" to record the time of each token
    
    fs << "---------------" << std::endl;
    
    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }

    autoregressive_seq_pos_ += input_tensors[0]->sequence();
}

// #define DYNAMIC
void QNNExecutor::execute(Net *net, vector<shared_ptr<Tensor>> input_tensors) {
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

    string typeName;
    if (executionType_ == PROMPT) {
        typeName = "Prompt_Graph.";
    } else if (executionType_ == AUTOREGRESSIVE) {
        typeName = "Autoregressive_Graph.";
    }

    for (auto Gid : flashGid) {
        net->subGraph()[typeName + std::to_string(Gid)]->reflashInput(net->tensors());
    }

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = typeName + std::to_string(i);
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
        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];
#endif

        g->reshape();
        g->setUpTensors();

        std::cout << "QNNGraph forward begin" << std::endl;

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
