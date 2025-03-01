#include <cassert>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <omp.h>
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
// for print graph execute time
#define QNN_EXECUTE_TIME 1

BackendType QNNExecutor::graphOffloadRule(BackendType expectedBackend, int graphIndex) {
    if (expectedBackend != MLLM_CPU && expectedBackend != MLLM_QNN) {
        return MLLM_CPU;
    } else {
        return expectedBackend;
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

    static_cast<QNNBackend *>(net->backends()[MLLM_QNN].get())->setDataLoader(data_loader_);

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = graphNamingRule(i);
        auto &g = net->subGraph()[name];

        g->reshape();
        g->setUpTensors();
    }
    auto ex_time_end = mllm_time_us();

    ex_time_start = mllm_time_us();

    // execute all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        uint64_t t_start = mllm_time_us();

        auto &g = net->subGraph()[graphNamingRule(i)];
        result_ = g->forward();

        uint64_t t_end = mllm_time_us();
#ifdef QNN_EXECUTE_TIME
        if (g->device() == MLLM_CPU) {
            std::cout << " TIME of CPU Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        } else {
            std::cout << " TIME of QNN Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        }
#endif
    }

    ex_time_end = mllm_time_us();

    // free all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        auto &g = net->subGraph()[graphNamingRule(i)];
        if (g->device() != MLLM_QNN) {
            continue;
        }
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free();
    }
    // use the second graph to free all context is OK.
    {
        auto &g = net->subGraph()[graphNamingRule(1)];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->allFree();
    }

    if (input_tensors[0]->sequence() == 1) {
        auto token_run_time = (ex_time_end - ex_time_start) / 1000.0F;
        run_time_.push_back(token_run_time);
    }
    std::cout << "prefill time: " << (ex_time_end - ex_time_start) / 1000.0F << "ms" << std::endl;
}

void QNNPipelineExecutor::warmup(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    auto ex_time_start = mllm_time_us();
    // input will be split into chunks and execute in pipeline
    int chunk_num = (input_tensors[0]->sequence() + chunk_size_ - 1) / chunk_size_;
    // we suppose the tensor(s) of input_tensors is the only one or all have the same seq length
    for (int i = 0; i < input_tensors.size(); ++i) {
        if (i != 0) {
            assert(input_tensors[i]->sequence() == input_tensors[i - 1]->sequence());
        }
    }

    // create a new tensor for each chunk
    // (chunk_num, vector<shared_ptr<Tensor>>(input_tensors.size()));
    chunked_tensors_list.resize(chunk_num, vector<shared_ptr<Tensor>>(input_tensors.size()));

    if (!isSetup_) {
        bool init = false;
        bool reshape = false;
        // split the tensor in chunks
        for (int i = 0; i < chunk_num; ++i) {
            // for all inputs in input_tensors
            auto &chunked_tensors = chunked_tensors_list[i];
            for (int j = 0; j < input_tensors.size(); ++j) {
                chunked_tensors[j] = std::make_shared<Tensor>();
                chunked_tensors[j]->setBackend(net->backends()[BackendType::MLLM_CPU].get());
                chunked_tensors[j]->reshape(1, 1, chunk_size_, 1);
                chunked_tensors[j]->setName(net->inputNames()[j]);
                // use shallowCopyFrom for each chunk to avoid memcpy
                chunked_tensors[j]->shallowCopyFrom(input_tensors[j].get(), false, {0, 0, i * chunk_size_, 0});
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

        PRINT_MEMORY_USAGE("before setup all graph");

        static_cast<QNNBackend *>(net->backends()[MLLM_QNN].get())->setDataLoader(data_loader_);

        for (int i = 0; i < (int)net->subGraph().size(); ++i) {
            string name = graphNamingRule(i);
            auto &g = net->subGraph()[name];

            g->reshape();
            g->setUpTensors();
        }
        isSetup_ = true;
    }
    auto ex_time_end = mllm_time_us();
    std::cout << "warmup done for " << (ex_time_end - ex_time_start) / 1000000.0 << "s" << std::endl;
}

void QNNPipelineExecutor::run(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensors) {
    auto ex_time_start = mllm_time_us();

    // input will be split into chunks and execute in pipeline
    int chunk_num = (input_tensors[0]->sequence() + chunk_size_ - 1) / chunk_size_;
    // we suppose the tensor(s) of input_tensors is the only one or all have the same seq length
    for (int i = 0; i < input_tensors.size(); ++i) {
        if (i != 0) {
            assert(input_tensors[i]->sequence() == input_tensors[i - 1]->sequence());
        }
    }

    if (!isSetup_) {
        warmup(ctx, net, input_tensors);
    }

    auto ex_time_end = mllm_time_us();

    ex_time_start = mllm_time_us();

    // execute all graphs here
    vector<shared_ptr<Tensor>> chunked_result_list;

    std::function<void(int, int)> executeFunc = [&](int chunk_id, int graphIdx) {
        int i = graphIdx - chunk_id;
        if (i < 0 || i >= (int)net->subGraph().size()) {
            return;
        }

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

        auto &g = net->subGraph()[name];
        if (chunk_id != 0 && g->device() == MLLM_CPU) {
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

        auto t_end = mllm_time_us();

#ifdef QNN_EXECUTE_TIME
        if (g->device() == MLLM_CPU) {
            std::cout << " TIME of CPU Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        } else {
            std::cout << " TIME of QNN Graph " << i << ": " << (t_end - t_start) / 1000.0F << "ms, End at " << (t_end - ex_time_start) / 1000.f << std::endl;
        }
#endif

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
                                          chunk_size_ * chunk_num,
                                          chunked_result_list[tid]->dimension());
                    result_[tid]->alloc();
                }
            }

            // move the result to the final result
            for (int tid = 0; tid < chunked_result_list.size(); ++tid) {
                auto &result_tensor = chunked_result_list[tid];

                memcpy(result_[tid]->ptrAt<float>(0, 0, chunk_size_ * chunk_id, 0), result_tensor->hostPtr<float>(), result_tensor->count() * sizeof(float));
            }
        }
    };

    omp_set_max_active_levels(3);
    // based on chunk_num, execute it every 2 chunk in pipeline
    for (int chunk_id = 0; chunk_id < chunk_num / 2; ++chunk_id) {
        // for every two chunk, start at chunk_id * 2 to avoid no execute for
        for (int i = chunk_id * 2; i < (int)net->subGraph().size() + chunk_id * 2 + 5; ++i) {
#pragma omp parallel for num_threads(2)
            for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
                executeFunc(chunk_id * 2 + pair_idx, i - pair_idx * 4);
            }
#pragma omp barrier
#ifdef QNN_EXECUTE_TIME
            std::cout << "---------------------------" << std::endl;
#endif
        }
    }
    // the last chunk if there is odd chunks
    if (chunk_num % 2 == 1) {
        for (int i = 0; i < (int)net->subGraph().size(); ++i) {
            executeFunc(chunk_num - 1, i);
        }
    }

    ex_time_end = mllm_time_us();

    // free all graphs here
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        auto expectedBackend = ctx->sub_backend_[i];
        if (expectedBackend == MLLM_CPU || i == 0) { // use CPU graph and CPU backend for embedding, based on specific subgraph split
            continue;
        }

        string name = graphNamingRule(i);
        auto &g = net->subGraph()[name];
        auto *qnn_graph = dynamic_cast<QNNGraph *>(g.get());
        qnn_graph->free();
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

} // namespace mllm
