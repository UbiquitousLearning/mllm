//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef TOKENIZATION_LLAMA_HPP
#define TOKENIZATION_LLAMA_HPP

#include "tokenizers/BPE/Bpe.hpp"

using namespace mllm;

class LLaMATokenizer final {
    BPETokenizer *tokenizer;

    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }
    bool bos_ = true;

public:
    explicit LLaMATokenizer(const std::string &vocab_file, bool bos = true) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);
        bos_ = bos;
    }
    Tensor tokenize(std::string &text, int str_i = 0) const {
        auto tokens_id = vector<token_id_t>();
        tokenizer->tokenize(text, tokens_id, bos_);
        return BPETokenizer::tokens2Input(tokens_id);
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

#endif // TOKENIZATION_LLAMA_HPP
