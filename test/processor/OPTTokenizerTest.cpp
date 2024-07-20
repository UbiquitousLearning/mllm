//
// Created by 咸的鱼 on 2024/2/23.
//
#include "gtest/gtest.h"
#include "TokenizorTest.hpp"
#include "tokenizers/BPE/Bpe.hpp"
TEST_F(TokenizerTest, OPTTokenizerTest) {
    GTEST_SKIP();
    auto bpe = new mllm::BPETokenizer("./vocab_opt.mllm");
    std::unordered_map<string,unsigned> merge_rank;
    auto merge_file = std::ifstream("./merges.txt");
    std::string line;
    unsigned rank=0;
    while (std::getline(merge_file, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0]=='#'){
            continue;
        }
        merge_rank[line]=rank;
        rank++;
    }
    bpe->setMergeRank(merge_rank);
    std::cout<<bpe->getVocabSize()<<std::endl;
    vector<mllm::token_id_t> tokens={};
    string text="Hello, world!";
    text = mllm::Tokenizer::replaceString(text,' ',"Ġ");
    bpe->setSpecialToken("</s>","");
    bpe->tokenize(text,tokens,true);

    for (auto token:tokens){
        std::cout<<token<< " ";
    }


}
