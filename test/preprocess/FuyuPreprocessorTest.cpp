#include <gtest/gtest.h>
//
// Created by 咸的鱼 on 2023/12/8.
//
#include "TokenizorTest.hpp"
#include "preprocess/FuyuPreProcess.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
using namespace mllm;

TEST_F(TokenizerTest, FuyuPreprocessorTest) {
    auto unigram = UnigramTokenizer("vocab_uni.mllm");
    auto preprocessor = FuyuPreProcess(&unigram);
    preprocessor.PreProcessImages({"bus.png"});
    preprocessor.Process("a coco-style image captioning model");
    auto input_ids = preprocessor.image_input_ids_;
    auto attention_mask = preprocessor.attention_mask_;
    auto image_patches = preprocessor.image_patches_;
    std::cout<<"Input Id"<<std::endl;
    for (auto id : input_ids) {
        for (auto id_ : id) {
            std::cout << id_ << " ";
        }
        std::cout << std::endl;
    }
    std::cout<<"Attention Mask"<<std::endl;
    for (auto id : attention_mask) {
        for (auto id_ : id) {
            std::cout << id_ << " ";
        }
        std::cout << std::endl;
    }
    std::cout<<"Image Patches"<<std::endl;
//     for (auto id : image_patches) {
//         for (auto id_ : id) {
//             for (auto id__ : id_) {
//                     std::cout << id__ << " ";
//             }
//             std::cout << std::endl;
//
//
//         }
//         std::cout << std::endl;
//     }
    std::cout << "Image Patches Size" << std::endl;
    std::cout << image_patches.size() << std::endl;
    std::cout << image_patches[0].size() << std::endl;
    std::cout << image_patches[0][0].size() << std::endl;
}