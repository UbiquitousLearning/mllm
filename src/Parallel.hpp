#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "tokenizers/Tokenizer.hpp"
#include <string>

namespace mllm {

class ChunkPipeline {
    int real_seq_length, chunk_size, chunk_num;
    vector<Tensor> chunked_tensors;

public:
    ChunkPipeline(int real_seq_length = 4, int chunk_size = 64) :
        real_seq_length(real_seq_length), chunk_size(chunk_size) {
        const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
        chunk_num = seq_length_padding / chunk_size;
        chunked_tensors.resize(chunk_num);
    }

    Tensor run(Tensor &input_tensor, LlmTextGeneratorOpts &opt, Tokenizer &tokenizer, Module &model, bool &isSwitched) {
        for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
            chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU]);
            chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
            chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
            chunked_tensors[chunk_id].setName("input-chunk-" + std::to_string(chunk_id));
            chunked_tensors[chunk_id].deepCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

            model.generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                if (!isSwitched && chunk_id == 0 && static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->isStageSwitching()) {
                    // turn off switching at the first chunk of following inputs
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                    isSwitched = true;
                }
                auto out_string = tokenizer.detokenize({out_token});
                auto [not_end, output_string] = tokenizer.postprocess(out_string);
                if (!not_end) { return false; }
                if (chunk_id == chunk_num - 1) { // print the output of the last chunk
                    std::cout << output_string << std::flush;
                }
                return true;
            });
            Module::isFirstChunk = false;
        }
        return chunked_tensors.back();
    }
};

} // namespace mllm

#endif // PARALLEL_HPP