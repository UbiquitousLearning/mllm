//
// Created by 咸的鱼 on 2023/12/21.
//

#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP
#include "tokenizers/Tokenizer.hpp"
namespace mllm {
class PreProcessor {
protected:
    mllm::Tokenizer *tokenizer_;
public:
    explicit PreProcessor(mllm::Tokenizer *tokenizer) : tokenizer_(tokenizer) {}
};
}

#endif //PREPROCESS_HPP
