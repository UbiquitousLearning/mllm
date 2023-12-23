//
// Created by 咸的鱼 on 2023/12/23.
//
#include "PreProcess.hpp"

#include <cassert>
#ifndef  STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif
#include "PreProcess.hpp"
#include "imageHelper/stb_image_resize2.h"
using namespace mllm;
std::vector<ImageInfo> PreProcessor::PadImages(std::vector<ImageInfo> &images, int height, int width, size_t patch_width, size_t patch_height, float pad_value, PaddingType padding_type, bool free_source) {
    assert(padding_type == PaddingType::CONSTANT);
    auto padded_images = std::vector<ImageInfo>();
    for (auto image : images) {
        // if (image.height == height && image.width == width && image.channels == 3) {
        //     padded_images.emplace_back(image.data, image.width, image.height, image.channels, image.original_width, image.original_height);
        //
        //     continue;
        // }
        if (image.height % patch_height == 0 && image.width % patch_width == 0) {
            padded_images.emplace_back(image.data, image.width, image.height, image.channels, image.original_width, image.original_height);
            continue;
        }
        auto height_ = height;
        auto width_ = width;
        height_ = (image.height / patch_height + 1) * patch_height;
        width_ = (image.width / patch_width + 1) * patch_width;
        auto padded_image =  new float[height_ * width_ * image.channels]{pad_value};
        std::fill(padded_image, padded_image + height_ * width_ * image.channels, pad_value);
        for (int i = 0; i < image.height; i++) {
            for (int j = 0; j < image.width; j++) {
                for (int k = 0; k < image.channels; k++) {
                    padded_image[(i * width_ + j) * image.channels + k] = image.data[(i * image.width + j) * image.channels + k];
                }
            }
        }
        padded_images.emplace_back(padded_image , width_, height_, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        // delete &images;
        images.clear();
    }
    return padded_images;
}

std::vector<ImageInfo> PreProcessor::ResizeImages(std::vector<ImageInfo> &images, int height, int width, ResampleType resample_type, bool free_source) {
    assert(resample_type == ResampleType::BILINEAR);
    stbir_filter filter = stbir_filter::STBIR_FILTER_DEFAULT;
    switch (resample_type) {
    case ResampleType::BILINEAR: filter = stbir_filter::STBIR_FILTER_TRIANGLE;
        break;
    default: std::cerr << "Not implemented Reshape Filter yet" << std::endl;
    // exit(-1);
        filter = stbir_filter::STBIR_FILTER_DEFAULT;
    }
    auto resized_images = std::vector<ImageInfo>();
    for (auto image : images) {
        if (image.height <= height && image.width <= width && image.channels == 3) {
            resized_images.emplace_back(image.data, image.width, image.height, image.channels, image.original_width, image.original_height);
            continue;
        }
        auto height_ = height;
        auto width_ = width;
        auto height_ratio = static_cast<float>(height) / image.height;
        auto width_ratio = static_cast<float>(width) / image.width;
        auto ratio = std::min(height_ratio, width_ratio);
        height_ = static_cast<int>(image.height * ratio);
        width_ = static_cast<int>(image.width * ratio);
        auto resized_image = new float[height_ * width_ * image.channels];
        stbir_resize(image.data, image.width, image.height, 0, resized_image, width_, height_, 0, stbir_pixel_layout::STBIR_RGB, stbir_datatype::STBIR_TYPE_FLOAT, stbir_edge::STBIR_EDGE_CLAMP, filter);
        resized_images.emplace_back(resized_image, width_, height_, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        // delete &images;
        // images.clear();
        images.clear();
    }

    return resized_images;
}

std::vector<ImageInfo> PreProcessor::NormalizeImages(std::vector<ImageInfo> &images, float mean, float std, bool free_source) {
    auto normalized_images = std::vector<ImageInfo>();
    for (auto image : images) {
        auto normalized_image = new float[image.width * image.height * image.channels];
        for (int i = 0; i < image.width * image.height * image.channels; i++) {
            normalized_image[i] = (image.data[i] - mean) / std;
        }
        normalized_images.emplace_back(normalized_image, image.width, image.height, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        // delete &images;
        images.clear();
    }
    return normalized_images;
}
