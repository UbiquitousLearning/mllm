/**
 * @file tokenization_Yi.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_YI_HPP
#define TOKENIZATION_YI_HPP

#include "tokenizers/BPE/Bpe.hpp"

using namespace mllm;

class YiTokenizer final {
    BPETokenizer *tokenizer;

    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

public:
    explicit YiTokenizer(const std::string &vocab_file) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);
    }

    Tensor tokenize(std::string &text, int str_i = 0) const {
        if (text[0] != ' ') {
            text = ' ' + text;
        }
        auto tokens_id = vector<token_id_t>();
        tokenizer->tokenize(text, tokens_id, false);
        if (str_i > 0) {
            tokens_id[0] = 13;
        }
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

#endif // !TOKENIZATION_YI_HPP
