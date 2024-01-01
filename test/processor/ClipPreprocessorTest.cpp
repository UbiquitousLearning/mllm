//
// Created by 咸的鱼 on 2023/12/31.
//

#include "gtest/gtest.h"
#include "TokenizorTest.hpp"
#include "processor/ClipPreProcess.hpp"
#include "tokenizers/BPE/Bpe.hpp"
TEST_F(TokenizerTest, ClipPreProcess) {
   auto tokenizer = new mllm::BPETokenizer("./vit_vocab.mllm");
    vector<mllm::token_id_t> tokens={};
    tokenizer->tokenize(" a photo of a cat",tokens,false);
    for (auto token:tokens){
        std::cout<<token<< " ";
    }
}