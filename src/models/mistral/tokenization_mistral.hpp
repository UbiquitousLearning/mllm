/**
 * @file tokenization_mistral.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief Mistral tokenization method
 * @version 0.1
 * @date 2024-05-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_MISTRAL_HPP
#define TOKENIZATION_MISTRAL_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <algorithm>
#include <regex>

using namespace mllm;

class MistralTokenizer final {
public:
    explicit MistralTokenizer(const std::string &vocab_file) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);
    }

    ~MistralTokenizer() {
        delete tokenizer;
    }

    Tensor tokenize(std::string &text) const {
        auto newText = token_start + token_user_o + " " + text + token_user_c + token_end;
        auto tokens_id = vector<token_id_t>();
        tokenizer->tokenize(text, tokens_id, false);
        return BPETokenizer::tokens2Input(tokens_id);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) {
        assert(result.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(result.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        std::vector<float> scores;
        int _dims = result.dimension();
        int _seq = result.sequence() - 1;
        for (int i = 0; i < _dims; ++i) {
            auto value = result.dataAt<float>(0, 0, _seq, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        auto text = tokenizer->detokenize({token_idx});
        return make_pair(text, token_idx);
    }

private:
    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

    BPETokenizer *tokenizer;

public:
    token_id_t eos_id = 2, bos_id = 1;
    std::string token_user_o = "[INST]", token_user_c = "[/INST]";
    std::string token_start = "<s>", token_end = "</s>";
    std::string token_unkonw = "<unk>";
};

#endif //! TOKENIZATION_MISTRAL_HPP