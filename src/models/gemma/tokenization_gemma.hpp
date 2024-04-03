/**
 * @file tokenization_gemma.hpp
 * @author Chenghua Wang (chenghua.wang@gmail.com)
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_GEMMA_HPP
#define TOKENIZATION_GEMMA_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <algorithm>

using namespace mllm;

class GemmaTokenizer final {
public:
private:
    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

    BPETokenizer *tokenizer;
};

#endif //! TOKENIZATION_GEMMA_HPP
