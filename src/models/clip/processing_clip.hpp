//
// Created by Rongjie Yi on 2024/2/19 0004.
//

#ifndef TOKENIZATION_CLIP_HPP
#define TOKENIZATION_CLIP_HPP

#include <utility>

#include "processor/ClipPreProcess.hpp"
#include "tokenizers/BPE/Bpe.hpp"

#include <numeric>

using namespace mllm;

class ClipProcessor final {
    Tensor img2Tensor(vector<vector<vector<float>>> img, string name = "input", BackendType type = MLLM_CPU) {
        int channel = img.size();
        int height = img[0].size();
        int width = img[0][0].size();
        Tensor tensor1(1, height, channel, width, Module::backends[type], true);
        tensor1.setName(std::move(name));
        tensor1.status() = TENSOR_STATIC_INIT;
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
    vector<float> softmax(const vector<float>& scores) {
        vector<float> exps;
        float max_val = *max_element(scores.begin(), scores.end());
        for (float score : scores) {
            exps.push_back(exp(score - max_val));
        }
        float sum_exps = accumulate(exps.begin(), exps.end(), 0.0f);
        for (float& exp : exps) {
            exp /= sum_exps;
        }
        return exps;
    }

    BPETokenizer *tokenizer;
    ClipPreProcessor *clip_processor;

public:
    explicit ClipProcessor(const string &vocab_path, const string &merges_path) {
        tokenizer = new BPETokenizer(vocab_path);
        std::unordered_map<string, unsigned> merge_rank;
        auto merge_file = std::ifstream(merges_path);
        std::string line;
        unsigned rank = 0;
        while (std::getline(merge_file, line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] == '#') {
                continue;
            }
            merge_rank[line] = rank;
            rank++;
        }
        tokenizer->setMergeRank(merge_rank);
        tokenizer->setSpecialToken("<|startoftext|>", "<|endoftext|>");
        clip_processor = new ClipPreProcessor(tokenizer);
    }

    std::array<Tensor, 2> process(vector<string> in_strs , string img_path, int hw = 224,
                   string img_name = "input_vision", string text_name = "input_text", BackendType type = MLLM_CPU) {

        // vector<string> in_strs = {"a photo of a cat", "a photo of a dog"};
        auto tokens_ids = vector<vector<token_id_t>>();
        for (auto in_str : in_strs) {
            vector<mllm::token_id_t> tokens_id = {};
            tokenizer->tokenize(in_str, tokens_id, true, true, "</w>");
            tokens_ids.push_back(tokens_id);
        }
        clip_processor->PreProcessImages({std::move(img_path)}, hw, hw);
        auto images = clip_processor->pixel_values_[0];

        return {Tokenizer::tokens2Input(tokens_ids), img2Tensor(images, std::move(img_name))};
    }
    vector<float> postProcess(const Tensor &result) {
        vector<float> scores;
        for (int i = 0; i < result.batch(); ++i) {
            auto value = result.dataAt<float>(i, 0, 0, 0);
            scores.push_back(value);
        }
        auto token_idx =  softmax(scores);
        // for (auto prob : token_idx) {
        //     std::cout << prob << "  ";
        // }
        return token_idx;
    }
};

#endif // TOKENIZATION_CLIP_HPP
