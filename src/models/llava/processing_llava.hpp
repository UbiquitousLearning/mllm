//
// Created by Rongjie Yi on 24-3-8.
//

#ifndef PROCESSING_LLAVA_HPP
#define PROCESSING_LLAVA_HPP
#include "tokenizers/BPE/Bpe.hpp"
#include "processor/ClipPreProcess.hpp"
#include <numeric>
#include <utility>

using namespace mllm;

class LLaVAProcessor final {
    Tensor img2Tensor(vector<vector<vector<float>>> img, string name = "input", BackendType type = MLLM_CPU) {
        int channel = img.size();
        int height = img[0].size();
        int width = img[0][0].size();
        Tensor tensor1(1, height, channel, width, Backend::global_backends[type], true);
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int h = 0; h < height; ++h) {
            for (int c = 0; c < channel; ++c) {
                for (int w = 0; w < width; ++w) {
                    tensor1.setDataAt<float>(0, h, c, w, img[c][h][w]);
                }
            }
        }
        return tensor1;
    }
    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        unsigned int maxIndex = 0;
        float maxValue = scores[0];
        for (size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > maxValue) {
                maxIndex = i;
                maxValue = scores[i];
            }
        }
        return maxIndex;
    }

    BPETokenizer *tokenizer;
    ClipPreProcessor *clip_processor;

public:
    explicit LLaVAProcessor(const string &vocab_path, const string &merges_path) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_path);
        std::unordered_map<string,unsigned> merge_rank;
        auto merge_file = std::ifstream(merges_path);
        std::string line;
        unsigned rank=0;
        while (std::getline(merge_file, line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0]=='#'){
                continue;
            }
            merge_rank[line]=rank;
            rank++;
        }
        tokenizer->setMergeRank(merge_rank);
    }

    std::array<Tensor, 2> process(string text, string img_path, int hw = 336,
                                  string img_name = "input_vision", string text_name = "input_text", BackendType type = MLLM_CPU) {
        auto tokens_ids = vector<vector<token_id_t>>();
        if (text[0] != ' ') {
            text = ' ' + text;
        }
        vector<mllm::token_id_t> tokens_id = {};
        tokenizer->tokenize(BPETokenizer::replaceString(text, ' ', "▁"), tokens_id, {"<image>", "<pad>", "\n"});
        tokens_ids.push_back(tokens_id);
        clip_processor = new ClipPreProcessor(tokenizer, hw, hw);
        clip_processor->PreProcessImages({std::move(img_path)}, hw, hw);
        auto images = clip_processor->pixel_values_[0];

        return {Tokenizer::tokens2Input(tokens_ids, std::move(text_name)), img2Tensor(images, std::move(img_name))};
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) {
        assert(result.batch() == 1);
        assert(result.head() == 1);
        vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        return {tokenizer->detokenize({token_idx}), token_idx};
    }
};

#endif // PROCESSING_LLAVA_HPP
