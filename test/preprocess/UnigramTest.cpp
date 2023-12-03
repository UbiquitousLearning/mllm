//
// Created by 咸的鱼 on 2023/12/3.
//
#include "TokenizorTest.hpp"
#include "gtest/gtest.h"
#include "tokenizers/Unigram/Unigram.hpp"
TEST_F(TokenizerTest, test) {
    auto tokenizer = std::make_shared<mllm::UnigramTokenizer>("vocab_uni.mllm");
    std::vector<mllm::token_id_t> ids;
    tokenizer->setSpecialToken("|ENDOFTEXT|");
      tokenizer->tokenize("_Hello world", ids, true);
    for (auto id : ids) {
        std::cout << id << " ";
    }
    auto result = tokenizer->detokenize(ids);
    std::cout << result << std::endl;


}