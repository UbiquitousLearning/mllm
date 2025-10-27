//
// Created by Xiang Li on 2023/12/29.
//

#include "ClipPreProcess.hpp"
#include <cstdlib>
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb/stb_image.h"

namespace mllm {
void ClipPreProcessor::Process(const std::string &text) {
    auto token_id = vector<token_id_t>();
    tokenizer_->tokenize(text, token_id, false);
    input_ids_.push_back(token_id);
}

void ClipPreProcessor::PreProcessImages(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length) {
    auto imageinfos = vector<ImageInfo>();
    for (int i = 0; i < images.size(); i++) {
        int width, height, channels;
        auto data = stbi_load_from_memory(images[i], image_length[i], &width, &height, &channels, 3);
        if (data == nullptr) {
            MLLM_LOG_ERROR_STREAM << "Error: Failed to load image from memory." << std::endl;
            exit(-1);
        }
        float *f32_data = nullptr;
        if (do_rescale_) {
            f32_data = PreProcessor::RescaleImage(data, scale_, width * height * channels);
            stbi_image_free(data);
        } else {
            f32_data = PreProcessor::RescaleImage(data, 1.0F, width * height * channels);
            stbi_image_free(data);
        }
        imageinfos.emplace_back(ImageInfo(f32_data, width, height, channels));
    }
    if (do_resize_) {
        imageinfos = PreProcessor::ResizeImages(imageinfos, height_, width_, false, true, shortest);
    }
    // std::cout<<imageinfos[0].height<< imageinfos[0].width <<std::endl;
    // Use height_ or crop_size?
    imageinfos = PreProcessor::CenterCropImages(imageinfos, height_, width_, 0, true);

    if (do_normalize_) {
        imageinfos = PreProcessor::NormalizeImages(imageinfos, mean_, std_);
    }
    // todo: Optimize this!
    for (auto &imageinfo : imageinfos) {
        auto pixel_values = vector<vector<vector<float>>>();
        for (int k = 0; k < imageinfo.channels; k++) {
            auto channel = vector<vector<float>>();
            for (int i = 0; i < imageinfo.height; i++) {
                auto row = vector<float>();
                for (int j = 0; j < imageinfo.width; j++) {
                    row.push_back(imageinfo.get_whc_pixel(i * imageinfo.width + j + k * imageinfo.width * imageinfo.height));
                }
                channel.push_back(row);
            }
            pixel_values.push_back(channel);
        }

        pixel_values_.push_back(pixel_values);
    }
}

void ClipPreProcessor::PreProcessImages(const std::vector<std::string> &images_path) {
    assert(height_ > 0 && width_ > 0);
    auto image_data = std::vector<uint8_t *>();
    auto image_length = std::vector<size_t>();
    for (const auto &i : images_path) {
        // read all file contents
        std::ifstream file(i, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            MLLM_LOG_ERROR_STREAM << "Cannot open file: " << i << std::endl;
            exit(-1);
        }
        auto size = file.tellg();
        auto data = new uint8_t[size];
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char *>(data), size);
        file.close();
        image_data.emplace_back(data);
        image_length.emplace_back(size);
    }
    PreProcessImages(image_data, image_length);
}

void ClipPreProcessor::Img2Tensor(Backend *bn, shared_ptr<Tensor> input_tensor, vector<vector<vector<float>>> img) {
    int channel = img.size();
    int height = img[0].size();
    int width = img[0][0].size();
    input_tensor->setBackend(bn);
    input_tensor->reshape(1, height, channel, width);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int h = 0; h < height; ++h) {
        for (int c = 0; c < channel; ++c) {
            for (int w = 0; w < width; ++w) {
                input_tensor->setDataAt<float>(0, h, c, w, img[c][h][w]);
            }
        }
    }
}

// vector<Tensor> ClipPreProcessor::process(std::string &text, vector<string> image) {
//     // auto tokens_ids = vector<vector<token_id_t>>();
//     // for (auto in_str : text) {
//     //     vector<mllm::token_id_t> tokens_id = {};
//     //     tokenizer_->tokenize(in_str, tokens_id, true, true, "</w>");
//     //     tokens_ids.push_back(tokens_id);
//     // }
//     // this->PreProcessImages({std::move(image)}, 224, 224);
//     // auto images = this->pixel_values_[0];

//     // return {Tokenizer::tokens2Input(tokens_ids), img2Tensor(images, std::move("input_vision"))};
//     std::cout << "TODO clip::process" << std::endl;
//     exit(-1);
//     return {};
// }
// std::pair<std::string, unsigned> ClipPreProcessor::detokenize(Tensor &result) {
//     return tokenizer_->detokenize(result);
// }

// std::pair<bool, std::string> ClipPreProcessor::postprocess(std::string &text) {
//     // size_t pos = 0;
//     // std::string from = "▁";
//     // std::string to = " ";
//     // while ((pos = text.find(from, pos)) != std::string::npos) {
//     //     text.replace(pos, from.length(), to);
//     //     pos += to.length();
//     // }
//     // if (text == "|ENDOFTEXT|") return {false, ""};
//     return {true, text};
// }

} // namespace mllm
