//
// Created by 咸的鱼 on 2024/1/23.
//
#include "gtest/gtest.h"
#include "TokenizorTest.hpp"
#include "tokenizers/BPE/Bpe.hpp"
using namespace  mllm;
TEST_F(TokenizerTest, LLavaTest) {
    auto tokenizer = new BPETokenizer("tmp.mllm");
    std::vector<vector<string>> in_imgs = {
        {"./assets/australia.jpg"}};
    vector<string> in_strs = {
        "<image> USER: What's the content of the image? ASSISTANT:"};
    vector<mllm::token_id_t> tokens_id = {};
    std::unordered_map<string,unsigned> merge_rank;
    auto merge_file = std::ifstream("./tmp.mllm.merges.txt");
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
    tokenizer->setMergeRank(merge_rank);
    tokenizer->tokenize(    mllm::BPETokenizer::replaceString(in_strs[0],' ',"▁"), tokens_id, true);
    for (auto id : tokens_id) {
        std::cout<<id<<" ";
    }
    std::cout<<tokenizer->detokenize(tokens_id);

    // for (int inId = 0; inId < in_strs.size(); ++inId) {
    //     auto in_str = in_strs[0];
    //     auto in_img = in_imgs[0];
    //
    //     auto tokens_ids = vector<vector<token_id_t>>();
    //     vector<mllm::token_id_t> tokens_id = {};
    //     tokenizer->tokenize(in_str, tokens_id, true);
    //     tokens_ids.push_back(tokens_id);
    //     for (auto id : tokens_id) {
    //         std::cout<<id<<" ";
    //     }
    //     std::cout<<std::endl;
    //
    // }
}