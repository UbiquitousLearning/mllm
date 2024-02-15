//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef TOKENIZATION_FUYU_HPP
#define TOKENIZATION_FUYU_HPP

#include "processor/FuyuPreProcess.hpp"
#include "tokenizers/Unigram/Unigram.hpp"

using namespace mllm;

class FuyuProcessor final {
    UnigramTokenizer *tokenizer;
    FuyuPreProcess *preprocessor;

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

    static Tensor patches2Tensor(vector<vector<vector<float>>> image_patches, string name = "input", BackendType type = MLLM_CPU) {
        int batch = 0;
        int seq = 0;
        int dims = 0;
        if (!image_patches.empty()) {
            batch = image_patches.size();
            seq = image_patches[0].size();
            dims = image_patches[0][0].size();
        }
        Tensor tensor1(batch, 1, seq, dims, Module::backends[type], true);
        tensor1.setName(name);
        tensor1.status() = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < seq; ++j) {
                for (int k = 0; k < dims; ++k) {
                    tensor1.setDataAt<float>(i, 0, j, k, image_patches[i][j][k]);
                }
            }
        }
        return tensor1;
    }

    static Tensor patchIdx2Tensor(vector<vector<int>> image_patches_indices, string name = "input", BackendType type = MLLM_CPU) {
        int batch = 0;
        int seq = 0;
        if (!image_patches_indices.empty()) {
            batch = image_patches_indices.size();
            seq = image_patches_indices[0].size();
        }
        Tensor tensor1(batch, 1, seq, 1, Module::backends[type], true);
        tensor1.setName(name);
        tensor1.status() = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < seq; ++j) {
                tensor1.setDataAt<float>(i, 0, j, 0, image_patches_indices[i][j]);
            }
        }
        return tensor1;
    }

public:
    explicit FuyuProcessor(const std::string &vocab_file) {
        tokenizer = new UnigramTokenizer(vocab_file);
    }

    std::array<Tensor, 3> process(std::string &text, vector<string> image) {
        preprocessor = new FuyuPreProcess(tokenizer);
        preprocessor->images_.clear();
        preprocessor->image_input_ids_.clear();
        preprocessor->image_patches_indices_.clear();
        preprocessor->image_patches_.clear();
        preprocessor->PreProcessImages(image);
        preprocessor->Process(text);
        auto input_ids = preprocessor->image_input_ids_;
        auto image_patches_indices = preprocessor->image_patches_indices_;
        auto image_patches = preprocessor->image_patches_;
        if (input_ids.empty()) {
            input_ids = preprocessor->text_ids_;
        }
        std::array<Tensor, 3> result = {UnigramTokenizer::tokens2Input(input_ids[0], "input_ids"),
                                        patches2Tensor(image_patches, "image_patches"),
                                        patchIdx2Tensor(image_patches_indices, "image_patches_indices")};
        return result;
    }

    Tensor tokenize(std::string &text) const {
        if (text[0] != ' ') {
            text = ' ' + text;
        }
        auto tokens_id = vector<token_id_t>();
        tokenizer->tokenize(text, tokens_id, true);
        return UnigramTokenizer::tokens2Input(tokens_id);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }
    std::pair<std::string, unsigned> detokenize(const Tensor &result) {
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

#endif // TOKENIZATION_FUYU_HPP
