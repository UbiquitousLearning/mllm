/**
 * @file tokenization_dclm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-09-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_DCLM_HPP
#define TOKENIZATION_DCLM_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include "tokenizers/Tokenizer.hpp"
#include <algorithm>

using namespace mllm;

using namespace mllm;

class DCLMTokenizer final {
    BPETokenizer *tokenizer;
    std::unordered_map<std::string, unsigned> merge_rank;

    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }
    bool bos_ = true;

public:
    explicit DCLMTokenizer(const std::string &vocab_file, const std::string &merge_file) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);
        std::ifstream merge(merge_file);
        std::string line;
        unsigned rank = 0;
        while (std::getline(merge, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            merge_rank[line] = rank++;
        }
        tokenizer->setMergeRank(merge_rank);
        tokenizer->setSpecialToken("<|endoftext|>", "<|endoftext|>", "<|endoftext|>");
    }
    Tensor tokenize(std::string &text) const {
        text = Tokenizer::replaceString(text, ' ', "Ġ");
        std::vector<token_id_t> tokens_id;
        tokenizer->tokenize(text, tokens_id, false);
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

#endif //! TOKENIZATION_DCLM_HPP