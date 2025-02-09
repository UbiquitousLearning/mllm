#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include "Backend.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "tokenizers/Tokenizer.hpp"
#include <memory>

namespace mllm {

class ChunkPipeline {
    int real_seq_length, chunk_size, chunk_num;
    vector<shared_ptr<Tensor>> chunked_tensors;

public:
    ChunkPipeline(int real_seq_length = 4, int chunk_size = 64) :
        real_seq_length(real_seq_length), chunk_size(chunk_size) {
        const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
        chunk_num = seq_length_padding / chunk_size;
    }

    shared_ptr<Tensor> run(Tensor &input_tensor, LlmTextGeneratorOpts &opt, Tokenizer &tokenizer, Module &model, bool &isSwitched) {
        const int num_graph = Tracer::model_.size();
        Tensor::tensor_status = TENSOR_STATIC_READY;
        std::cout << "num_graph: " << num_graph << std::endl;

        for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
            chunked_tensors.push_back(std::make_shared<Tensor>(Backend::global_backends[MLLM_CPU]));
            chunked_tensors[chunk_id]->setTtype(INPUT_TENSOR);
            chunked_tensors[chunk_id]->setName(input_tensor.name());
            chunked_tensors[chunk_id]->reshape(1, 1, chunk_size, 1);
            chunked_tensors[chunk_id]->shallowCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});
        }

        std::function<void(int, int)> executeFunc = [&](int chunk_id, int graphIdx) {
            int i = graphIdx - chunk_id;
            // out of range
            if (i < 0 || i >= num_graph) {
                return;
            }
            // only the last chunk need to execute the last graph
            if(i == num_graph - 1 && chunk_id != chunk_num - 1) {
                return;
            }
            // before the first graph, need to refresh the input tensor
            if (i == 0) {
                Tracer::refleshInputTensor({chunked_tensors[chunk_id]});
            }

            auto graph_start = mllm_time_us();
            auto &graph = Tracer::model_[i];
            graph->Forward({}, {chunk_id});
            auto graph_end = mllm_time_us();
            std::cout << "chunk_id: " << chunk_id << ", graphIdx: " << i << ", graph time: " << (graph_end - graph_start) / 1000.0F << "ms" << std::endl;
        };
        auto start_t = mllm_time_us();
        omp_set_max_active_levels(3);
        for (int chunk_id = 0; chunk_id < chunk_num / 2; ++chunk_id) {
            // for every two chunk, start at chunk_id * 2 to avoid no execute for
            for (int i = chunk_id * 2; i < num_graph + chunk_id * 2 + 5; ++i) {
#pragma omp parallel for num_threads(2)
                for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
                    executeFunc(chunk_id * 2 + pair_idx, i - pair_idx * 4);
                }
#pragma omp barrier
                std::cout << "---------------------------" << std::endl;
            }
        }
        auto end_t = mllm_time_us();
        std::cout << "time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;

        auto postProcessing = [&](shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result, int real_seq_length) -> unsigned int {
            assert(result->batch() == 1);
            assert(result->head() == 1);
            out_result->reshape(1, 1, 1, 1);
            out_result->alloc();
            vector<float> scores;
            for (int i = 0; i < result->dimension(); ++i) {
                auto value = result->dataAt<float>(0, 0, real_seq_length % chunk_size - 1, i);
                scores.push_back(value);
            }
            auto arg_max = [&]() -> unsigned int {
                return std::max_element(scores.begin(), scores.end()) - scores.begin();
            };
            auto token_idx = arg_max();
            out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
            return token_idx;
        };

        auto cpuModulePtr = std::dynamic_pointer_cast<CPUModuleWrapper>(Tracer::model_.back());
        auto result = cpuModulePtr->result();
        auto token_idx = postProcessing(result[0], chunked_tensors.back(), real_seq_length);
        auto out_string = tokenizer.detokenize({token_idx});
        std::cout << out_string << std::flush;
        return chunked_tensors.back();
    }
};

} // namespace mllm

#endif // PARALLEL_HPP