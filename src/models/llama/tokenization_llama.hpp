//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef TOKENIZATION_LLAMA_HPP
#define TOKENIZATION_LLAMA_HPP

#include "tokenizers/BPE/Bpe.hpp"

using namespace mllm;


class LLaMATokenizer final {
    BPETokenizer* tokenizer;


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
public:
    explicit LLaMATokenizer(const std::string &vocab_file) {
        tokenizer = new BPETokenizer(vocab_file);
    }
    Tensor tokenize(std::string &text) const {
        if (text[0] != ' ') {
            text = ' ' + text;
        }
        auto tokens_id = vector<token_id_t>();
        tokenizer->tokenize(text, tokens_id, true);
        // if (str_i > 0) {
        //     tokens_id[0] = 13;
        // }
        return BPETokenizer::tokens2Input(tokens_id);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(const Tensor& result) {
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

#endif //TOKENIZATION_LLAMA_HPP
