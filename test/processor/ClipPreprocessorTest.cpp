//
// Created by Xiang Li on 2023/12/31.
//

#include "gtest/gtest.h"
#include "TokenizorTest.hpp"
#include "processor/ClipPreProcess.hpp"
#include "tokenizers/BPE/Bpe.hpp"
TEST_F(TokenizerTest, ClipPreProcess) {
   auto tokenizer = new mllm::BPETokenizer("../vocab/clip_vocab.mllm");
    //read merges.txt and split it into merge_rank
    std::unordered_map<string,unsigned> merge_rank;
    auto merge_file = std::ifstream("../vocab/clip_merges.txt");
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
    std::cout<<tokenizer->getVocabSize()<<std::endl;
    vector<mllm::token_id_t> tokens={};
    string text="a photo of a cat";
    // text = mllm::Tokenizer::replaceString(text,' ',"</w>");
    tokenizer->setSpecialToken("<|startoftext|>","<|endoftext|>");
    tokenizer->tokenize(text,tokens,true);
    for (auto token:tokens){
        std::cout<<token<< " ";
    }
}
TEST_F(TokenizerTest,Clip) {
    auto tokenizer = new mllm::BPETokenizer("../vocab/clip_vocab.mllm");
    mllm::ClipPreProcessor* clip = new mllm::ClipPreProcessor(tokenizer);
    clip->PreProcessImages({"../assets/bus.png"});
    auto images = clip->pixel_values_[0];
    std::cout << "size: " << images.size()<<" " <<images[0].size()  << " " << images[0][0].size() << std::endl;
    // for (auto row:images){
    //     for (auto pixel:row){
    //         for (auto channel:pixel){
    //             std::cout<<channel<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }
}