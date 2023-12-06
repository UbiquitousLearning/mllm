//
// Created by 咸的鱼 on 2023/12/4.
//

#include "FuyuPreProcess.hpp"

#include <iostream>

namespace mllm {
std::vector<FourDVector> FuyuPreProcess::PreProcessImages(const std::vector<vector<uint8_t>> &images, int height, int width, bool do_pad, bool do_resize) {
    assert(height > 0 && width > 0);

    // if (do_resize) {
    //     // Not implemented yet
    //     std::cerr << "Resize not implemented yet" << std::endl;
    //     exit(-1);
    // }
    auto images_ = std::vector<ImageInfo>();
    images_.resize(images.size());
    for (auto image : images) {
        int width_, height_, channels_;
        unsigned char *data = stbi_load_from_memory(image.data(), image.size(), &width, &height, &channels_, 0);
        if (data == nullptr) {
            std::cerr << "load image failed" << std::endl;
            exit(-1);
        }
        float *float_data = new float[width * height * channels_];
        for (int i = 0; i < width * height * channels_; i++) {
            float_data[i] = data[i] / 255.0;
        }

        images_.emplace_back(float_data, width_, height_, channels_);
    }
    auto image_patches = std::vector<FourDVector>();
    // TODO: PAD images
    auto padded_images = PadImages(images_, height, width);
}

std::vector<ImageInfo> FuyuPreProcess::PadImages(const std::vector<ImageInfo> &images, int height, int width, float pad_value, PaddingType padding_type) {
    assert(padding_type == PaddingType::CONSTANT);
    auto padded_images = std::vector<ImageInfo>();
    for (auto image : images) {
        if(image.height == height && image.width == width && image.channels == 3){
            padded_images.emplace_back(image.data,image.width,image.height,image.channels,image.original_width,image.original_height);

            continue;
        }
        auto padded_image = std::vector<float>();
        padded_image.resize(height * width * image.channels, pad_value);
        for (int i = 0; i < image.height; i++) {
            for (int j = 0; j < image.width; j++) {
                for (int k = 0; k < image.channels; k++) {
                    padded_image[(i * width + j) * image.channels + k] = image.data[(i * image.width + j) * image.channels + k];
                }
            }
        }
        padded_images.emplace_back(padded_image.data(), width, height, image.channels,image.original_width,image.original_height);
        free(image.data);
        image.data = nullptr;
    }
    return padded_images;
}

std::vector<ImageInfo> FuyuPreProcess::ResizeImages(const std::vector<ImageInfo> &images, int height, int width, ResampleType resample_type) {
    assert(resample_type == ResampleType::BILINEAR);
    stbir_filter filter = stbir_filter::STBIR_FILTER_DEFAULT;
    switch (resample_type) {
        case ResampleType::BILINEAR:
            filter = stbir_filter::STBIR_FILTER_TRIANGLE;
            break;
        default:
            std::cerr << "Not implemented Reshape Filter yet" << std::endl;
            // exit(-1);
            filter = stbir_filter::STBIR_FILTER_DEFAULT;

    }
    auto resized_images = std::vector<ImageInfo>();
    for (auto image : images) {
        auto resized_image = std::vector<float>();
        resized_image.resize(height * width * image.channels);
        stbir_resize(image.data, image.width, image.height, 0, resized_image.data(), width, height, 0, stbir_pixel_layout::STBIR_RGB, stbir_datatype::STBIR_TYPE_FLOAT, stbir_edge::STBIR_EDGE_CLAMP,filter);
        resized_images.emplace_back(resized_image.data(), width, height, image.channels,image.original_width,image.original_height);
        free(image.data);
        image.data = nullptr;
    }
    return resized_images;
}
} // mllm
