#include <gtest/gtest.h>
//
// Created by Xiang Li on 2023/12/8.
//
#include "TokenizorTest.hpp"
#include "processor/FuyuPreProcess.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
using namespace mllm;

TEST_F(TokenizerTest, FuyuPreprocessorTest) {
    // GTEST_SKIP();
    auto unigram = UnigramTokenizer("../vocab/fuyu_vocab.mllm");

    auto preprocessor = new FuyuPreProcess(&unigram);
    preprocessor->PreProcessImages({"../assets/bus.png"});
    preprocessor->Process("Generate a coco-style caption.\n");
    auto input_ids = preprocessor->image_input_ids_;
    auto image_patches_indices = preprocessor->image_patches_indices_;
    auto image_patches = preprocessor->image_patches_;
    std::cout<< "Input Id Size "<<input_ids[0].size()<<std::endl;
    std::cout<< "Image Patches Indices Size "<<image_patches_indices[0].size()<<std::endl;

    std::cout<<"Input Id"<<std::endl;
    // for (auto id : input_ids) {
    //     for (auto id_ : id) {
    //         std::cout << id_ << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout<<"Attention Mask"<<std::endl;
    // for (auto id : attention_mask) {
    //     for (auto id_ : id) {
    //         std::cout << id_ << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout<<"Image Patches"<<std::endl;
    //  for (auto id : image_patches) {
    //      for (const auto& id_ : id) {
    //          for (const auto idx : id_) {
    //                  std::cout << idx << " ";
    //          }
    //          std::cout << std::endl;
    //
    //
    //      }
    //      std::cout << std::endl;
    //  }
    std::cout << "Image Patches Size" << std::endl;
    std::cout << image_patches.size() << std::endl;
    std::cout << image_patches[0].size() << std::endl;
    std::cout << image_patches[0][0].size() << std::endl;
}
TEST_F(TokenizerTest, FuyuPatchImages) {
    GTEST_SKIP();

    auto height = 420;
    auto width = 640;
    auto channels = 3;
    auto image = (float *) malloc(height * width * channels * sizeof(float));
    for (int i = 0; i < height * width * channels; i++) {
        image[i] = i;
    }
    auto image_info = ImageInfo(image, width, height, channels);
    auto patches = mllm::FuyuPreProcess::PatchImages({image_info},30,30);
    std::cout << "Patches Size" << std::endl;
    std::cout << patches.size() << std::endl;
    std::cout << patches[0].size() << std::endl;
    auto patch = patches[0];
    // for (float i : patch) {
    for (int i = 0; i < patch.size(); i++) {
        std::cout << patch[i] << " ";
       if (i>2700) break;
        }
    }
// we only check the first pair

