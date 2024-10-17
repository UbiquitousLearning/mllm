/**
 * @file tokenization_gemma.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-04-03
 *
 * @details check https://github.com/google/gemma_pytorch/blob/main/gemma/tokenizer.py
 * The gemma's tokenizer used a subset of the SentencePiece tokenizer
 * see tech-report: https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_GEMMA_HPP
#define TOKENIZATION_GEMMA_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <algorithm>
#include <regex>

using namespace mllm;

class GemmaTokenizer final {
public:
    explicit GemmaTokenizer(const std::string &vocab_file) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);
    }

    ~GemmaTokenizer() {
        delete tokenizer;
    }

    Tensor tokenize(std::string &text) const {
        // replace all blanck to '_'
        std::string new_text = BPETokenizer::replaceString(text, ' ', "▁");

        // Returns a tokenized string. The Gemma tokenizer never adds a prefix space
        auto tokens_id = vector<token_id_t>();
        tokenizer->tokenize(new_text, tokens_id, false);

        // insert <bos>
        tokens_id.insert(tokens_id.begin(), bos_id);
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
    token_id_t pad_id = 0, eos_id = 1, bos_id = 2, unk_id = 3;
};

#endif //! TOKENIZATION_GEMMA_HPP
