/**
 * @file tokenization_qwen.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-04-29
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_QWEN_HPP
#define TOKENIZATION_QWEN_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <algorithm>
#include <regex>

using namespace mllm;
class QWenTokenizer final {
public:
    explicit QWenTokenizer(const std::string &vocab_file) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);
    }

    ~QWenTokenizer() {
        delete tokenizer;
    }

    Tensor tokenize(std::string &text, int str_i = 0) const {
        std::vector<token_id_t> tokens_id;
        tokenizer->tokenize(text, tokens_id, /*insert bos*/ false, /*byte-level fallback*/ true, /*end symbol*/ "");
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
    token_id_t eos_id = 151643, bos_id = 151643, im_start_id = 151644, im_end_id = 151645;
};

#endif //! TOKENIZATION_QWEN_HPP