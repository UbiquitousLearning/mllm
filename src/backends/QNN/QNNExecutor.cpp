#include <cassert>
#include <csignal>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>
#include "QNNBackend.hpp"
#include "QNNGraph.hpp"
#include "Timing.hpp"
#include "QNNExecutor.hpp"
#include "MemInspect.hpp"
#include "Types.hpp"
#include "express/ExpressBase.hpp"

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
    PRINT_MEMORY_USAGE("before setup all graph");
    shared_ptr<Backend> a = net->backends()[MLLM_QNN];
    // cast to QNNBackend*
    // auto *qnn = dynamic_cast<QNNBackend *>(a.get());
    // qnn->swapMemManager();
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];

        // cast graph to QNNGraph
        // TODO: if this implementation is used, the setUpTensors(string) should be merged to Graph
        // the qnn_graph below is where we cast the Graph to QNNGraph
        if (i == 0) {
            std::cout << "=======setup cpu graph " << i << std::endl;
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
        uint64_t t_start = mllm_time_us();
        if (i == 0) {
            std::cout << "======= cpu graph execute" << i << std::endl;
            string name = typeName + std::to_string(i);
            auto &g = net->subGraph()[name];
            result_ = g->forward();
        } else {
            std::cout << "=======qnn graph execute" << i << std::endl;
            string name = typeName + std::to_string(i);
            auto &g = net->subGraph()[name];
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            result_ = qnn_graph->forward(name);
            uint64_t t_end = mllm_time_us();
            std::cout << "graph forward " << (t_end - t_start) / 1000.0F << "ms" << std::endl;
            PRINT_MEMORY_USAGE((string("execute graph: ") + std::to_string(i)).c_str());
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
        if (i == 0) continue;
        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free(name);
    }

    // use the first graph to free all context is OK.
    {
        string name = typeName + std::to_string(1);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->allFree();
    }

    // open file "AR_latency.txt" to record the time of each token

    fs << "---------------" << std::endl;

    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }

    autoregressive_seq_pos_ += input_tensors[0]->sequence();
}

void QNNExecutor::run(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensors) {
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
    PRINT_MEMORY_USAGE("before setup all graph");
    shared_ptr<Backend> a = net->backends()[MLLM_QNN];

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];

        // cast graph to QNNGraph
        // TODO: if this implementation is used, the setUpTensors(string) should be merged to Graph
        // the qnn_graph below is where we cast the Graph to QNNGraph
        auto expectedBackend = ctx->sub_backend_[i];
        if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
            std::cout << "=======setup cpu graph " << i << std::endl;
            g->reshape();
            g->setUpTensors();
        } else {
            std::cout << "Graph" << i << " expectedBackend: " << expectedBackend << std::endl;
            if (i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
                std::cout << "=======setup cpu graph " << i << std::endl;
                g->reshape();
                g->setUpTensors();
            } else {
                std::cout << "=======setup qnn graph " << i << std::endl;
                auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
                g->reshape();
                // if ( autoregressive_seq_pos_ % 32 == 31 || autoregressive_seq_pos_ == 0) {
                // g->setUpTensors();
                qnn_graph->setUpTensors(name);
            }
        }

        if (false) {
            if (i < (int)net->subGraph().size() - 1) {
                g->freeTensors();
            }
            net->freeTensors(i);
        }
    }
    auto ex_time_end = mllm_time_us();

    fs << "setup all graph" << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;

    ex_time_start = mllm_time_us();

    // execute all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        uint64_t t_start = mllm_time_us();

        auto expectedBackend = ctx->sub_backend_[i];
        if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
            std::cout << "=======execute cpu graph " << i << std::endl;
            string name = typeName + std::to_string(i);
            auto &g = net->subGraph()[name];
            result_ = g->forward();
        } else {
            std::cout << "Graph" << i << " expectedBackend: " << expectedBackend << std::endl;
            if (i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
                std::cout << "=======execute cpu graph " << i << std::endl;
                string name = typeName + std::to_string(i);
                auto &g = net->subGraph()[name];
                result_ = g->forward();
            } else {
                std::cout << "=======execute qnn graph " << i << std::endl;
                string name = typeName + std::to_string(i);
                auto &g = net->subGraph()[name];
                auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
                result_ = qnn_graph->forward(name);
                uint64_t t_end = mllm_time_us();
                std::cout << "graph forward " << (t_end - t_start) / 1000.0F << "ms" << std::endl;
                PRINT_MEMORY_USAGE((string("execute graph: ") + std::to_string(i)).c_str());
            }
        }
    }

    ex_time_end = mllm_time_us();
    fs << "execute all graph" << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;

    // free all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        auto expectedBackend = ctx->sub_backend_[i];
        if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
            continue;
        } else if (i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
            continue;
        }

        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free(name);
    }

    // use the first graph to free all context is OK.
    {
        string name = typeName + std::to_string(1);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->allFree();
    }

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

void QNNPipelineExecutor::run(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    bool init = false;
    bool reshape = false;

    // input will be split into chunks and execute in pipeline
    const int chunk_size = 32;
    int chunk_num = (input_tensors[0]->sequence() + chunk_size - 1) / chunk_size;
    // create a new tensor for each chunk
    vector<vector<shared_ptr<Tensor>>> chunked_tensors_list(chunk_num, vector<shared_ptr<Tensor>>(input_tensors.size()));
    // TODO: we suppose the input_tensors only have one tensor or in the same sequence
    for (int i = 0; i < input_tensors.size(); ++i) {
        if (i != 0) {
            assert(input_tensors[i]->sequence() == input_tensors[i - 1]->sequence());
        }
    }
    for (int i = 0; i < chunk_num; ++i) {
        // for all inputs in input_tensors
        auto &chunked_tensors = chunked_tensors_list[i];
        for (int j = 0; j < input_tensors.size(); ++j) {
            chunked_tensors[j] = std::make_shared<Tensor>();
            chunked_tensors[j]->setBackend(net->backends()[BackendType::MLLM_CPU].get());
            chunked_tensors[j]->reshape(1, 1, chunk_size, 1);
            chunked_tensors[j]->setName(net->inputNames()[j]);
            chunked_tensors[j]->alloc();
            memcpy(chunked_tensors[j]->hostPtr<float>(), input_tensors[j]->ptrAt<float>(0, 0, i * chunk_size, 0), chunk_size * sizeof(float) * input_tensors[j]->dimension());
        }
    }

    // uses
    checkReshape(init, reshape, chunked_tensors_list[0]);

    // set Input tensor
    vector<int> flashGid = {};
    for (int tid = 0; tid < net->inputNames().size(); ++tid) {
        auto input_name = net->inputNames()[tid];
        auto input_tensor = chunked_tensors_list[0][tid];
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
    PRINT_MEMORY_USAGE("before setup all graph");
    shared_ptr<Backend> a = net->backends()[MLLM_QNN];

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];

        // cast graph to QNNGraph
        // TODO: if this implementation is used, the setUpTensors(string) should be merged to Graph
        // the qnn_graph below is where we cast the Graph to QNNGraph
        auto expectedBackend = ctx->sub_backend_[i];
        if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
            std::cout << "=======setup cpu graph " << i << std::endl;
            g->reshape();
            g->setUpTensors();
        } else {
            std::cout << "Graph" << i << " expectedBackend: " << expectedBackend << std::endl;
            if (i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
                std::cout << "=======setup cpu graph " << i << std::endl;
                g->reshape();
                g->setUpTensors();
            } else {
                std::cout << "=======setup qnn graph " << i << std::endl;
                auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
                g->reshape();
                qnn_graph->setUpTensors(name);
            }
        }
    }
    auto ex_time_end = mllm_time_us();

    // creat subGraph().size() mutex to control the pipeline
    // TODO: if the chunk_num >> subGraph().size(), is there possibility when the latter chunk execute before the former chunk?
    std::vector<std::mutex> mutexes(net->subGraph().size());

    fs << "setup all graph" << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;

    ex_time_start = mllm_time_us();

    // execute all graphs here
    vector<shared_ptr<Tensor>> chunked_result_list;

    // wrap the execute loop in a thread
    std::function<void(int chunk_id)> chunkExecutionFunction = [&](int chunk_id) {
        std::cout << "======= chunk:" << chunk_id << " total graph " << net->subGraph().size() << std::endl;

        for (int i = 0; i < (int)net->subGraph().size(); ++i) {
            // lock the mutex of mutexes at i
            mutexes[i].lock();
            // ------------------------------

            if (i == 0) {
                // update the input tensor for each chunk
                for (int tid = 0; tid < net->inputNames().size(); ++tid) {
                    auto input_name = net->inputNames()[tid];
                    auto input_tensor = chunked_tensors_list[chunk_id][tid];
                    net->tensors()[input_name] = input_tensor;
                }
            }

            uint64_t t_start = mllm_time_us();
            auto expectedBackend = ctx->sub_backend_[i];
            std::cout << "Graph" << i << " expectedBackend: " << expectedBackend << std::endl;
            if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
                std::cout << "======= chunk:" << chunk_id << " execute cpu graph " << i << std::endl;
                string name = typeName + std::to_string(i);
                auto &g = net->subGraph()[name];
                chunked_result_list = g->forward();
            } else {
                if (i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
                    std::cout << "======= chunk:" << chunk_id << " execute cpu graph " << i << std::endl;
                    string name = typeName + std::to_string(i);
                    auto &g = net->subGraph()[name];

                    result_ = g->forward();
                    chunked_result_list = g->forward();
                } else {
                    std::cout << "======= chunk:" << chunk_id << " execute qnn graph " << i << std::endl;
                    string name = typeName + std::to_string(i);
                    auto &g = net->subGraph()[name];
                    auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
                    chunked_result_list = qnn_graph->forward(name);
                }
            }
            uint64_t t_end = mllm_time_us();
            std::cout << "graph forward " << (t_end - t_start) / 1000.0F << "ms" << std::endl;
            PRINT_MEMORY_USAGE((string("execute graph: ") + std::to_string(i)).c_str());

            // if it is the last graph, move the result to the final result
            if (i == (int)net->subGraph().size() - 1) {
                result_.resize(chunked_result_list.size());
                if (chunk_id == 0) { // reshape the result tensor when first chunk is executed
                    for (int tid = 0; tid < chunked_result_list.size(); ++tid) {
                        result_[tid] = std::make_shared<Tensor>();
                        result_[tid]->setBackend(net->backends()[BackendType::MLLM_CPU].get());
                        result_[tid]->reshape(chunked_result_list[tid]->batch(),
                                              chunked_result_list[tid]->head(),
                                              chunk_size * chunk_num,
                                              chunked_result_list[tid]->dimension());
                        result_[tid]->alloc();
                    }
                }
                // move the result to the final result
                // TODO: segmentation fault
                for (int tid = 0; tid < chunked_result_list.size(); ++tid) {
                    auto &result_tensor = chunked_result_list[tid];
                    memcpy(result_[tid]->ptrAt<float>(0, 0, chunk_size * chunk_id, 0), result_tensor->hostPtr<float>(), result_tensor->count() * sizeof(float));
                }
            }

            // ------------------------------
            // unlock the mutex of mutexes at i
            mutexes[i].unlock();
        }
    };

    // wrap the thread pool execution in a function and await the thread pool to finish
    std::function executeFunction = [&]() {
        // create graph_num threads of executeFunction
        // use thread pool to manage the threads
        ThreadPool thread_pool(net->subGraph().size());
        for (int i = 0; i < chunk_num; ++i) {
            thread_pool.enqueue(std::bind(chunkExecutionFunction, i));
        }
    };
    executeFunction();

    std::cout << "CHECK RESULT" << std::endl;
    std::cout << result_.size();

    ex_time_end = mllm_time_us();
    fs << "execute all graph" << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;

    // in pipeline execute, don't free the graph
    // free all graphs here
    /*
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        auto expectedBackend = ctx->sub_backend_[i];
        if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
            continue;
        } else if (i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
            continue;
        }

        string name = typeName + std::to_string(i);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free(name);
    }
    // use the first graph to free all context is OK.
    {
        string name = typeName + std::to_string(1);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->allFree();
    }
    */

    // open file "AR_latency.txt" to record the time of each token
    fs << "---------------" << std::endl;

    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }

    autoregressive_seq_pos_ += input_tensors[0]->sequence();
}

QNNPipelineExecutor::ThreadPool::ThreadPool(size_t num_threads) :
    stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(
            [this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock,
                                              [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty())
                            return;
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task();
                }
            });
    }
}

QNNPipelineExecutor::ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : workers_)
        worker.join();
}

void QNNPipelineExecutor::ThreadPool::enqueue(std::function<void()> f) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks_.emplace(std::move(f));
    }
    condition_.notify_one();
}

} // namespace mllm
