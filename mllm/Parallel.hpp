#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include "Backend.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "tokenizers/Tokenizer.hpp"
#include <memory>
#include <stdexcept>

namespace mllm {

class ChunkPipeline {
    int real_seq_length, chunk_size, chunk_num;
    vector<shared_ptr<Tensor>> chunked_tensors;

public:
    static int chunkCountFor(int sequence_length, int chunk_size) {
        if (sequence_length <= 0) {
            throw std::invalid_argument("sequence length must be positive");
        }
        if (chunk_size <= 0) {
            throw std::invalid_argument("chunk size must be positive");
        }
        return sequence_length / chunk_size + (sequence_length % chunk_size != 0);
    }

    static int paddedSequenceLengthFor(int sequence_length, int chunk_size) {
        return chunkCountFor(sequence_length, chunk_size) * chunk_size;
    }

    static int lastTokenIndexFor(int sequence_length, int chunk_size) {
        chunkCountFor(sequence_length, chunk_size); // validate arguments
        return (sequence_length - 1) % chunk_size;
    }

    ChunkPipeline(int real_seq_length = 4, int chunk_size = 64) :
        real_seq_length(real_seq_length), chunk_size(chunk_size),
        chunk_num(chunkCountFor(real_seq_length, chunk_size)) {
    }

    int chunkCount() const {
        return chunk_num;
    }

    int paddedSequenceLength() const {
        return chunk_num * chunk_size;
    }

    shared_ptr<Tensor> run(Tensor &input_tensor, LlmTextGeneratorOpts &opt,
                           Tokenizer &tokenizer, Module &model,
                           bool &isSwitched,
                           const vector<Tensor *> &clean_tensors = {},
                           bool parallel_chunks = true,
                           bool print_first_token = true) {
        if (input_tensor.backend() == nullptr) {
            throw std::invalid_argument("chunk pipeline input tensor must have a backend");
        }
        if (input_tensor.sequence() < paddedSequenceLength()) {
            throw std::invalid_argument("chunk pipeline input tensor is shorter than its padded sequence length");
        }

        chunked_tensors.clear();
        chunked_tensors.reserve(chunk_num);

        auto input_copy_sp = std::make_shared<Tensor>(input_tensor.backend());
        input_copy_sp->initFrom(input_tensor); // 初始化形状和数据类型
        input_copy_sp->copyFrom(input_tensor); // 深拷贝数据

        const int num_graph = Tracer::model_.size();
        if (num_graph == 0) {
            throw std::runtime_error("chunk pipeline requires a traced model");
        }
        Tensor::tensor_status = TENSOR_STATIC_READY;
        std::cout << "num_graph: " << num_graph << std::endl;

        for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
            chunked_tensors.push_back(std::make_shared<Tensor>(Backend::global_backends[MLLM_CPU].get()));
            chunked_tensors[chunk_id]->setTtype(INPUT_TENSOR);
            chunked_tensors[chunk_id]->setName(input_tensor.name());
            chunked_tensors[chunk_id]->reshape(1, 1, chunk_size, 1);
            chunked_tensors[chunk_id]->shallowCopyFrom(input_copy_sp, false, {0, 0, chunk_id * chunk_size, 0}, 1);
        }

        std::function<void(int, int)> executeFunc = [&](int chunk_id, int graphIdx) {
            int i = graphIdx - chunk_id;
            // out of range
            if (i < 0 || i >= num_graph) {
                return;
            }
            // only the last chunk need to execute the last graph
            if (i == num_graph - 1 && chunk_id != chunk_num - 1) {
                return;
            }
            // before the first graph, need to refresh the input tensor
            if (i == 0) {
                Tracer::refleshInputTensor({chunked_tensors[chunk_id]});
            }
#ifdef DEBUGPRINT
            auto graph_start = mllm_time_us();
#endif
            auto &graph = Tracer::model_[i];
            graph->Forward({}, {chunk_id});
#ifdef DEBUGPRINT
            auto graph_end = mllm_time_us();
            std::cout << "chunk_id: " << chunk_id << ", graphIdx: " << i << ", graph time: " << (graph_end - graph_start) / 1000.0F << "ms" << std::endl;
#endif
        };
        auto start_t = mllm_time_us();
        if (!parallel_chunks) {
            // HMX and QNN share the accelerator session. Execute one logical
            // chunk completely before starting the next to avoid interleaved
            // accelerator submissions and excess context residency.
            for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
                for (int graph_idx = 0; graph_idx < num_graph; ++graph_idx) {
                    executeFunc(chunk_id, chunk_id + graph_idx);
                }
            }
        } else {
            omp_set_max_active_levels(3);
            const int paired_chunk_count = chunk_num - chunk_num % 2;
            for (int pair_start = 0; pair_start < paired_chunk_count;
                 pair_start += 2) {
                // Run two chunks as a pipeline, with the second chunk five
                // graph stages behind.
                for (int i = pair_start; i < num_graph + pair_start + 5; ++i) {
#pragma omp parallel for num_threads(2)
                    for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
                        executeFunc(pair_start + pair_idx, i - pair_idx * 4);
                    }
#pragma omp barrier
                }
            }

            // Finish an unpaired final chunk sequentially.
            if (chunk_num % 2 != 0) {
                const int final_chunk_id = chunk_num - 1;
                for (int graph_idx = 0; graph_idx < num_graph; ++graph_idx) {
                    executeFunc(final_chunk_id, final_chunk_id + graph_idx);
                }
            }
        }
        auto end_t = mllm_time_us();
        std::cout << "prefill time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;

        auto postProcessing = [&](shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result, int real_seq_length) -> unsigned int {
            assert(result->batch() == 1);
            assert(result->head() == 1);
            out_result->reshape(1, 1, 1, 1);
            out_result->alloc();
            vector<float> scores;
            for (int i = 0; i < result->dimension(); ++i) {
                auto value = result->dataAt<float>(0, 0, lastTokenIndexFor(real_seq_length, chunk_size), i);
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
        if (cpuModulePtr == nullptr) {
            throw std::runtime_error("chunk pipeline expects the final traced graph to run on CPU");
        }
        auto result = cpuModulePtr->result();
        auto token_idx = postProcessing(result[0], chunked_tensors.back(), real_seq_length);
        if (print_first_token) {
            auto out_string = tokenizer.detokenize({token_idx});
            std::cout << out_string << std::flush;
        }

        for (auto tensor : clean_tensors) {
            tensor->reshape(0, 0, 0, 0);
            tensor->alloc();
        }

        return chunked_tensors.back();
    }
};

} // namespace mllm

#endif // PARALLEL_HPP
