#include <cassert>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <omp.h>
#include <thread>
#include <unordered_map>
#include <vector>
#include "QNNBackend.hpp"
#include "QNNGraph.hpp"
#include "Timing.hpp"
#include "QNNExecutor.hpp"
#include "memory/MemInspect.hpp"
#include "Types.hpp"
#include "express/ExpressBase.hpp"

namespace mllm {
BackendType QNNExecutor::graphOffloadRule(BackendType expectedBackend, int graphIndex) {
    if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
        return MLLM_CPU;
    } else {
        if (graphIndex == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
            return MLLM_CPU;
        } else {
            return MLLM_QNN;
        }
    }
}

void QNNExecutor::setup(Net *net) {
    mllm_time_init();

    uint64_t time_start = mllm_time_us();
    uint64_t time_end;

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        auto &g = net->subGraph()[graphNamingRule(i)];
        g->setUpOps(*data_loader_);
    }
    time_end = mllm_time_us();
    if (load_time_ == 0) {
        load_time_ = (time_end - time_start) / 1000.0F;
        std::cout << "Load model: " << load_time_ / 1000.0F << " s" << std::endl;
    }
}

void QNNExecutor::run(Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    std::cerr << "QNN Executor do not support this method" << std::endl;
    exit(1);
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

    for (auto Gid : flashGid) {
        net->subGraph()[graphNamingRule(Gid)]->reflashInput(net->tensors());
    }

    auto ex_time_start = mllm_time_us();
    PRINT_MEMORY_USAGE("before setup all graph");

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = graphNamingRule(i);
        auto &g = net->subGraph()[name];

        // cast graph to QNNGraph
        // the qnn_graph below is where we cast the Graph to QNNGraph
        auto expectedBackend = ctx->sub_backend_[i];
        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            g->reshape();
            g->setUpTensors();
        } else if (graphOffloadRule(expectedBackend, i) == MLLM_QNN) {
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            g->reshape();
            qnn_graph->setUpTensors(name);
        } else {
            std::cerr << "Backend Not Support" << std::endl;
            exit(1);
        }
    }
    auto ex_time_end = mllm_time_us();

    ex_time_start = mllm_time_us();

    // execute all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        uint64_t t_start = mllm_time_us();

        auto expectedBackend = ctx->sub_backend_[i];
        string name = graphNamingRule(i);
        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            auto &g = net->subGraph()[name];
            result_ = g->forward();
        } else if (graphOffloadRule(expectedBackend, i) == MLLM_QNN) {
            auto &g = net->subGraph()[name];
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            result_ = qnn_graph->forward(name);

            PRINT_MEMORY_USAGE((string("execute graph: ") + std::to_string(i)).c_str());
        } else {
            std::cerr << "Backend Not Support" << std::endl;
            exit(1);
        }

        uint64_t t_end = mllm_time_us();
#ifdef DEBUGPRINT
        std::cout << "graph forward " << (t_end - t_start) / 1000.0F << "ms " << i << std::endl;
#endif
        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            std::cout << " TIME of CPU Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        } else {
            std::cout << " TIME of QNN Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        }
    }

    ex_time_end = mllm_time_us();

    // free all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        auto expectedBackend = ctx->sub_backend_[i];
        if (graphOffloadRule(expectedBackend, i) != MLLM_QNN) {
            continue;
        }

        string name = graphNamingRule(i);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free(name);
    }
    // use the second graph to free all context is OK.
    {
        string name = graphNamingRule(1);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->allFree();
    }

    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }
    std::cout << "prefill time: " << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;
}

void QNNPipelineExecutor::run(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    bool init = false;
    bool reshape = false;

    // input will be split into chunks and execute in pipeline
    const int chunk_size = 32;
    int chunk_num = (input_tensors[0]->sequence() + chunk_size - 1) / chunk_size;
    // create a new tensor for each chunk
    vector<vector<shared_ptr<Tensor>>> chunked_tensors_list(chunk_num, vector<shared_ptr<Tensor>>(input_tensors.size()));
    // we suppose the tensor(s) of input_tensors is the only one or all have the same seq length
    for (int i = 0; i < input_tensors.size(); ++i) {
        if (i != 0) {
            assert(input_tensors[i]->sequence() == input_tensors[i - 1]->sequence());
        }
    }
    // split the tensor in chunks
    for (int i = 0; i < chunk_num; ++i) {
        // for all inputs in input_tensors
        auto &chunked_tensors = chunked_tensors_list[i];
        for (int j = 0; j < input_tensors.size(); ++j) {
            chunked_tensors[j] = std::make_shared<Tensor>();
            chunked_tensors[j]->setBackend(net->backends()[BackendType::MLLM_CPU].get());
            chunked_tensors[j]->reshape(1, 1, chunk_size, 1);
            chunked_tensors[j]->setName(net->inputNames()[j]);
            // use deepCopyFrom for each chunk to avoid memcpy
            chunked_tensors[j]->deepCopyFrom(input_tensors[j].get(), false, {0, 0, i * chunk_size, 0});
        }
    }

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

    for (auto Gid : flashGid) {
        net->subGraph()[graphNamingRule(Gid)]->reflashInput(net->tensors());
    }

    auto ex_time_start = mllm_time_us();
    PRINT_MEMORY_USAGE("before setup all graph");

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = graphNamingRule(i);
        auto &g = net->subGraph()[name];

        // cast graph to QNNGraph
        // the qnn_graph below is where we cast the Graph to QNNGraph
        auto expectedBackend = ctx->sub_backend_[i];

        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            g->reshape();
            g->setUpTensors();
        } else if (graphOffloadRule(expectedBackend, i) == MLLM_QNN) {
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            g->reshape();
            qnn_graph->setUpTensors(name);
        } else {
            std::cerr << "Backend Not Support" << std::endl;
            exit(1);
        }
    }
    auto ex_time_end = mllm_time_us();

    ex_time_start = mllm_time_us();

    // execute all graphs here
    // creat subGraph().size() mutex to control the pipeline
    std::vector<std::mutex> mutexes(net->subGraph().size());
    std::mutex chunk_mutex;
    std::mutex cpu_mutex;
    std::vector<int> graph_chunk_index(net->subGraph().size(), 0);
    vector<shared_ptr<Tensor>> chunked_result_list;

    std::condition_variable cv;
    omp_set_max_active_levels(2);
#pragma omp parallel for
    for (int i = 0; i < 2; ++i) {
        std::cout << "num of threads: " << omp_get_num_threads() << std::endl;
        std::cout << "thread: " << omp_get_thread_num() << std::endl;
        const int chunk_id = i;

        int core_id = sched_getcpu();
        std::cout << "executor core id: " << core_id << std::endl;
        for (int i = 0; i < (int)net->subGraph().size(); ++i) {
            // make sure chunks execute by order
            while (true) {
                chunk_mutex.lock();
                if (graph_chunk_index[i] == chunk_id) {
                    graph_chunk_index[i]++;
                    chunk_mutex.unlock();
                    break;
                } else {
                    chunk_mutex.unlock();
                    std::this_thread::yield();
                }
            }

            // std::unique_lock<std::mutex> lock(chunk_mutex);
            // cv.wait(lock, [&] { return graph_chunk_index[i] == chunk_id; });
            // graph_chunk_index[i]++;

            // make sure current graph is ready for this chunk
            // lock the mutex of mutexes at i
            // mutexes[i].lock();

            if (i == 0) {
                // update the input tensor for each chunk
                for (int tid = 0; tid < net->inputNames().size(); ++tid) {
                    auto input_name = net->inputNames()[tid];
                    auto input_tensor = chunked_tensors_list[chunk_id][tid];
                    unordered_map<string, shared_ptr<Tensor>> map;
                    map[input_name] = input_tensor;
                    string graphName = graphNamingRule(i);
                    net->subGraph()[graphName]->reflashInput(map);
                }
            }

            auto expectedBackend = ctx->sub_backend_[i];
            string name = graphNamingRule(i);
            auto t_start = mllm_time_us();
            if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
                // execute only one cpu graph at a time
                cpu_mutex.lock();
#ifdef DEBUGPRINT
                std::cout
                    << "chunk:" << chunk_id << " execute cpu graph " << i << std::endl;
#endif

                auto &g = net->subGraph()[name];
                if (chunk_id != 0) {
                    // cpu graph should reshape and setup for every chunk forward for KVCache op
                    g->reshape();
                    g->setUpTensors();
                }

                // only get the result at the last graph
                if (i == net->subGraph().size() - 1) {
                    chunked_result_list = g->forward();
                } else {
                    g->forward();
                }

                // execute only one cpu graph at a time
                cpu_mutex.unlock();
            } else if (graphOffloadRule(expectedBackend, i) == MLLM_QNN) {
#ifdef DEBUGPRINT
                std::cout << "chunk:" << chunk_id << " execute qnn graph " << i << std::endl;
#endif
                auto &g = net->subGraph()[name];
                auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
                qnn_graph->forward(name);

                // only get the result at the last graph
                if (i == net->subGraph().size() - 1) {
                    chunked_result_list = qnn_graph->forward(name);
                } else {
                    qnn_graph->forward(name);
                }
            } else {
                std::cerr << "Backend Not Support" << std::endl;
                exit(1);
            }

            auto t_end = mllm_time_us();

            if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
                std::cout << chunk_id << " TIME of CPU Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
            } else {
                std::cout << chunk_id << " TIME of QNN Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
            }

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
                for (int tid = 0; tid < chunked_result_list.size(); ++tid) {
                    auto &result_tensor = chunked_result_list[tid];

                    memcpy(result_[tid]->ptrAt<float>(0, 0, chunk_size * chunk_id, 0), result_tensor->hostPtr<float>(), result_tensor->count() * sizeof(float));
                }
            }

            // unlock the mutex of mutexes at i
            // mutexes[i].unlock();
        }
    }

    ex_time_end = mllm_time_us();

    // TODO: in pipeline execute, don't free the graph, error will occur in qnn memory manager deconstruct
    // free all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        auto expectedBackend = ctx->sub_backend_[i];
        if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
            continue;
        } else if (i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
            continue;
        }

        string name = graphNamingRule(i);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free(name);
    }
    // use the second graph to free all context is OK.
    {
        string name = graphNamingRule(1);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->allFree();
    }

    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }
    std::cout << "prefill time: " << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;
}

void QNNPipelineExecutor::runExp(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    bool init = false;
    bool reshape = false;

    // input will be split into chunks and execute in pipeline
    const int chunk_size = 32;
    int chunk_num = (input_tensors[0]->sequence() + chunk_size - 1) / chunk_size;
    // create a new tensor for each chunk
    vector<vector<shared_ptr<Tensor>>> chunked_tensors_list(chunk_num, vector<shared_ptr<Tensor>>(input_tensors.size()));
    // we suppose the tensor(s) of input_tensors is the only one or all have the same seq length
    for (int i = 0; i < input_tensors.size(); ++i) {
        if (i != 0) {
            assert(input_tensors[i]->sequence() == input_tensors[i - 1]->sequence());
        }
    }
    // split the tensor in chunks
    for (int i = 0; i < chunk_num; ++i) {
        // for all inputs in input_tensors
        auto &chunked_tensors = chunked_tensors_list[i];
        for (int j = 0; j < input_tensors.size(); ++j) {
            chunked_tensors[j] = std::make_shared<Tensor>();
            chunked_tensors[j]->setBackend(net->backends()[BackendType::MLLM_CPU].get());
            chunked_tensors[j]->reshape(1, 1, chunk_size, 1);
            chunked_tensors[j]->setName(net->inputNames()[j]);
            // use deepCopyFrom for each chunk to avoid memcpy
            chunked_tensors[j]->deepCopyFrom(input_tensors[j].get(), false, {0, 0, i * chunk_size, 0});
        }
    }

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

    for (auto Gid : flashGid) {
        net->subGraph()[graphNamingRule(Gid)]->reflashInput(net->tensors());
    }

    auto ex_time_start = mllm_time_us();
    PRINT_MEMORY_USAGE("before setup all graph");

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = graphNamingRule(i);
        auto &g = net->subGraph()[name];

        // cast graph to QNNGraph
        // the qnn_graph below is where we cast the Graph to QNNGraph
        auto expectedBackend = ctx->sub_backend_[i];

        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            g->reshape();
            g->setUpTensors();
        } else if (graphOffloadRule(expectedBackend, i) == MLLM_QNN) {
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            g->reshape();
            qnn_graph->setUpTensors(name);
        } else {
            std::cerr << "Backend Not Support" << std::endl;
            exit(1);
        }
    }
    auto ex_time_end = mllm_time_us();

    ex_time_start = mllm_time_us();

    // execute all graphs here
    // creat subGraph().size() mutex to control the pipeline
    std::vector<std::mutex> mutexes(net->subGraph().size());
    std::mutex chunk_mutex;
    std::mutex cpu_mutex;
    std::vector<int> graph_chunk_index(net->subGraph().size(), 0);
    vector<shared_ptr<Tensor>> chunked_result_list;

    std::condition_variable cv;

    std::function<void(int, int)> executeFunc = [&](int chunk_id, int graphIdx) {
        int i = chunk_id == 0 ? graphIdx : graphIdx - 1;
        // make sure chunks execute by order
        // while (true) {
        //     chunk_mutex.lock();
        //     if (graph_chunk_index[i] == chunk_id) {
        //         graph_chunk_index[i]++;
        //         chunk_mutex.unlock();
        //         break;
        //     } else {
        //         chunk_mutex.unlock();
        //         std::this_thread::yield();
        //     }
        // }

        if (i == 0) {
            // update the input tensor for each chunk
            for (int tid = 0; tid < net->inputNames().size(); ++tid) {
                auto input_name = net->inputNames()[tid];
                auto input_tensor = chunked_tensors_list[chunk_id][tid];
                unordered_map<string, shared_ptr<Tensor>> map;
                map[input_name] = input_tensor;
                string graphName = graphNamingRule(i);
                net->subGraph()[graphName]->reflashInput(map);
            }
        }

        auto expectedBackend = ctx->sub_backend_[i];
        string name = graphNamingRule(i);
        auto t_start = mllm_time_us();
        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            // execute only one cpu graph at a time
            cpu_mutex.lock();

            auto &g = net->subGraph()[name];
            if (chunk_id != 0) {
                // cpu graph should reshape and setup for every chunk forward for KVCache op
                g->reshape();
                g->setUpTensors();
            }

            // only get the result at the last graph
            if (i == net->subGraph().size() - 1) {
                chunked_result_list = g->forward();
            } else {
                g->forward();
            }

            // execute only one cpu graph at a time
            cpu_mutex.unlock();
        } else if (graphOffloadRule(expectedBackend, i) == MLLM_QNN) {
            auto &g = net->subGraph()[name];
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            qnn_graph->forward(name);

            // only get the result at the last graph
            if (i == net->subGraph().size() - 1) {
                chunked_result_list = qnn_graph->forward(name);
            } else {
                qnn_graph->forward(name);
            }
        } else {
            std::cerr << "Backend Not Support" << std::endl;
            exit(1);
        }

        auto t_end = mllm_time_us();

        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            std::cout << chunk_id << " TIME of CPU Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        } else {
            std::cout << chunk_id << " TIME of QNN Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        }

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
            for (int tid = 0; tid < chunked_result_list.size(); ++tid) {
                auto &result_tensor = chunked_result_list[tid];

                memcpy(result_[tid]->ptrAt<float>(0, 0, chunk_size * chunk_id, 0), result_tensor->hostPtr<float>(), result_tensor->count() * sizeof(float));
            }
        }
    };

    executeFunc(0, 0);
    omp_set_max_active_levels(3);
    for (int i = 1; i < (int)net->subGraph().size(); ++i) {
#pragma omp parallel for
        for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
            executeFunc(chunk_id, i);
        }
    }
    // the last graph of chunk 1
    {
        auto chunk_id = 1;
        auto i = net->subGraph().size() - 1;
        auto expectedBackend = ctx->sub_backend_[i];
        string name = graphNamingRule(i);
        auto t_start = mllm_time_us();
        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            // execute only one cpu graph at a time
            cpu_mutex.lock();

            auto &g = net->subGraph()[name];
            if (chunk_id != 0) {
                // cpu graph should reshape and setup for every chunk forward for KVCache op
                g->reshape();
                g->setUpTensors();
            }

            // only get the result at the last graph
            if (i == net->subGraph().size() - 1) {
                chunked_result_list = g->forward();
            } else {
                g->forward();
            }

            // execute only one cpu graph at a time
            cpu_mutex.unlock();
        } else if (graphOffloadRule(expectedBackend, i) == MLLM_QNN) {
            auto &g = net->subGraph()[name];
            auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
            qnn_graph->forward(name);

            // only get the result at the last graph
            if (i == net->subGraph().size() - 1) {
                chunked_result_list = qnn_graph->forward(name);
            } else {
                qnn_graph->forward(name);
            }
        } else {
            std::cerr << "Backend Not Support" << std::endl;
            exit(1);
        }

        auto t_end = mllm_time_us();

        if (graphOffloadRule(expectedBackend, i) == MLLM_CPU) {
            std::cout << chunk_id << " TIME of CPU Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        } else {
            std::cout << chunk_id << " TIME of QNN Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        }

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
            for (int tid = 0; tid < chunked_result_list.size(); ++tid) {
                auto &result_tensor = chunked_result_list[tid];

                memcpy(result_[tid]->ptrAt<float>(0, 0, chunk_size * chunk_id, 0), result_tensor->hostPtr<float>(), result_tensor->count() * sizeof(float));
            }
        }
    }

    ex_time_end = mllm_time_us();

    // TODO: in pipeline execute, don't free the graph, error will occur in qnn memory manager deconstruct
    // free all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        auto expectedBackend = ctx->sub_backend_[i];
        if (expectedBackend != MLLM_DEFAULT && expectedBackend != MLLM_QNN) {
            continue;
        } else if (i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
            continue;
        }

        string name = graphNamingRule(i);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free(name);
    }
    // use the second graph to free all context is OK.
    {
        string name = graphNamingRule(1);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->allFree();
    }

    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }
    std::cout << "prefill time: " << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;
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
