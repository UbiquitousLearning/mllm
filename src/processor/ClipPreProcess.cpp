//
// Created by 咸的鱼 on 2023/12/29.
//

#include "ClipPreProcess.hpp"
#ifndef  STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "imageHelper/stb_image.h"

namespace mllm {
void ClipProcessor::Process(const std::string &text) {
    auto token_id = vector<token_id_t>();
    tokenizer_->tokenize(text, token_id, false);
    input_ids_.push_back(token_id);
}

void ClipProcessor::PreProcessImages(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length) {
    auto imageinfos = vector<ImageInfo>();
    for (int i = 0; i < images.size(); i++) {
        int width, height, channels;
        auto data = stbi_load_from_memory(images[i], image_length[i], &width, &height, &channels, 3);
        if (data == nullptr) {
            std::cerr << "Error: Failed to load image from memory." << std::endl;
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
        imageinfos = PreProcessor::ResizeImages(imageinfos, height_, width_, false);
    }
    // Use height_ or crop_size?
    imageinfos = PreProcessor::CenterCropImages(imageinfos, height_, width_, 0, true);

    if (do_normalize_) {
        imageinfos = PreProcessor::NormalizeImages(imageinfos, mean_, std_);
    }
    //todo: Optimize this!
    for (auto &imageinfo : imageinfos) {
        auto pixel_values = vector<vector<vector<float>>>();
        for (int i = 0; i < imageinfo.height; i++) {
            auto row = vector<vector<float>>();
            for (int j = 0; j < imageinfo.width; j++) {
                auto pixel = vector<float>();
                for (int k = 0; k < imageinfo.channels; k++) {
                    pixel.push_back(imageinfo.get_whc_pixel(i * imageinfo.width * imageinfo.channels + j * imageinfo.channels + k));
                }
                row.push_back(pixel);
            }
            pixel_values.push_back(row);
        }
        pixel_values_.push_back(pixel_values);
    }
}
}
