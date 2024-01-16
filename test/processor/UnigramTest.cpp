//TokenizerTest.test
// Created by Xiang Li on 2023/12/3.
//
#include "TokenizorTest.hpp"
#include "gtest/gtest.h"
#include "tokenizers/Unigram/Unigram.hpp"
TEST_F(TokenizerTest, UnigramTest) {
    auto tokenizer = std::make_shared<mllm::UnigramTokenizer>("../vocab/fuyu_vocab.mllm");
    std::vector<mllm::token_id_t> ids;
    tokenizer->setSpecialToken("|ENDOFTEXT|");
    std::string text = "Generate a coco-style caption.\n";
     // normalization text
    // replace all " " to "▁"
    std::string text_ = "";
    for (auto &ch : text) {
        if (ch == ' ') {
            text_ += "▁";
        }else {
            text_ += ch;
        }
    }
    // std::replace(text.begin(), text.end(), ' ', L'▁');
    // prepend "_" to text
    std::string new_text = "▁" + std::string(text_);

    tokenizer->tokenize(new_text, ids, true);
    for (auto id : ids) {
        std::cout << id << " ";
    }
    auto result = tokenizer->detokenize(ids);
    std::cout << result << std::endl;


}