//
// Created by Guo Xiaoqiang on 2024/8/12.
//
#ifndef TOKENIZATION_PHI3_HPP
#define TOKENIZATION_PHI3_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <algorithm>
#include <regex>

using namespace mllm;

class Phi3Tokenizer final {
public:
    explicit Phi3Tokenizer(const std::string &vocab_file) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);
    }

    ~Phi3Tokenizer() {
        delete tokenizer;
    }

    Tensor tokenize(std::string &text, int str_i = 0) const {
        // replace all blanck to '_'
        std::string new_text = BPETokenizer::replaceString(text, ' ', "▁");

        auto tokens_id = vector<token_id_t>();
        tokenizer->tokenize(new_text, tokens_id, false);

        // chat template is as follows: <|user|>\n Question <|end|>\n <|assistant|>
        tokens_id.insert(tokens_id.begin(), user_id);
        tokens_id.insert(tokens_id.begin() + 1, 13);
        tokens_id.insert(tokens_id.end(), end_id);
        tokens_id.insert(tokens_id.end(), 13);
        tokens_id.insert(tokens_id.end(), assistant_id);

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
        text = std::regex_replace(text, std::regex("▁"), " ");
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
    token_id_t pad_id = 32000, eos_id = 32000, bos_id = 1, user_id = 32010, assistant_id = 32001, end_id = 32007;
};

#endif //! TOKENIZATION_PHI3_HPP
