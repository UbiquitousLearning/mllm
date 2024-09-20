//
// Created by Xiang Li on 2023/12/23.
//
#include "PreProcess.hpp"

#include <cassert>
#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif
#include "stb/stb_image_resize2.h"
using namespace mllm;

void PreProcessor::PreProcessImages(const std::vector<std::string> &images_path) {
    assert(height_ > 0 && width_ > 0);
    auto image_data = std::vector<uint8_t *>();
    auto image_length = std::vector<size_t>();
    for (const auto &i : images_path) {
        // read all file contents
        std::ifstream file(i, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << i << std::endl;
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

float *PreProcessor::RescaleImage(const uint8_t *data, const float scale, unsigned length) {
    auto *float_data = new float[length];
    for (int j = 0; j < length; j++) {
        float_data[j] = data[j] / scale;
    }
    return float_data;
}

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
        auto padded_image = new float[height_ * width_ * image.channels]{pad_value};
        std::fill(padded_image, padded_image + height_ * width_ * image.channels, pad_value);
        for (int i = 0; i < image.height; i++) {
            for (int j = 0; j < image.width; j++) {
                for (int k = 0; k < image.channels; k++) {
                    padded_image[(i * width_ + j) * image.channels + k] = image.data[(i * image.width + j) * image.channels + k];
                }
            }
        }
        padded_images.emplace_back(padded_image, width_, height_, image.channels, image.original_width, image.original_height);
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

std::vector<ImageInfo> PreProcessor::ResizeImages(std::vector<ImageInfo> &images, int height, int width, bool strict_size, bool fit, PreProcessor::ResizeFitEdge fit_edge, ResampleType resample_type, bool free_source) {
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
        if (image.height <= height && image.width <= width && image.channels == 3 && (strict_size == false && fit == false)) {
            resized_images.emplace_back(image.data, image.width, image.height, image.channels, image.original_width, image.original_height);
            continue;
        }
        auto height_ = height;
        auto width_ = width;
        if (strict_size == false && fit == false) {
            auto height_ratio = static_cast<float>(height) / image.height;
            auto width_ratio = static_cast<float>(width) / image.width;
            auto ratio = std::min(height_ratio, width_ratio);
            // rounded to int

            height_ = std::round(image.height * ratio);
            width_ = std::round(image.width * ratio);
        }
        if (fit && fit_edge != none) {
            auto shortest_ = std::min(image.height, image.width);
            auto longest_ = std::max(image.height, image.width);
            switch (fit_edge) {
            case shortest:
                longest_ = std::round(height * longest_ / shortest_);
                shortest_ = height;
                break;
            case longest:
                shortest_ = std::round(height * shortest_ / longest_);
                longest_ = height_;
                break;
            default:
                break;
            }
            if (image.height > image.width) {
                height_ = longest_;
                width_ = shortest_;
            } else {
                width_ = longest_;
                height_ = shortest_;
            }
        }
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

std::vector<ImageInfo> PreProcessor::CenterCropImages(std::vector<ImageInfo> &images, int height, int width, float pad, bool free_source) {
    auto cropped_images = std::vector<ImageInfo>();
    for (auto image : images) {
        if (image.height == height && image.width == width && image.channels == 3) {
            cropped_images.emplace_back(image.data, image.width, image.height, image.channels, image.original_width, image.original_height);
            continue;
        }
        auto height_ = height;
        auto width_ = width;
        auto height_offset = (image.height - height) / 2;
        auto width_offset = (image.width - width) / 2;
        auto top_index = height_offset;
        auto bottom_index = height_offset + height;
        auto left_index = width_offset;
        auto right_index = width_offset + width;
        auto cropped_image = new float[height * width * image.channels];
        if (top_index >= 0 && bottom_index <= image.height && left_index >= 0 && right_index <= image.width) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    for (int k = 0; k < image.channels; k++) {
                        cropped_image[(i * width + j) * image.channels + k] = image.data[((i + height_offset) * image.width + j + width_offset) * image.channels + k];
                    }
                }
            }
            cropped_images.emplace_back(cropped_image, width_, height_, image.channels, image.original_width, image.original_height);
        } else {
            auto top_pad = std::max(height_ - image.height, 0) / 2;
            auto bottom_pad = std::max(height_ - image.height, 0) + height_;
            auto left_pad = std::max(width_ - image.width, 0) / 2;
            auto right_pad = std::max(width_ - image.width, 0) + width_;
            for (int i = 0; i < height_; i++) {
                for (int j = 0; j < width_; j++) {
                    for (int k = 0; k < image.channels; k++) {
                        if (i < top_pad || i >= bottom_pad || j < left_pad || j >= right_pad) {
                            auto index = (i * width_ + j) * image.channels + k;
                            cropped_image[(i * width_ + j) * image.channels + k] = pad;
                        } else {
                            cropped_image[(i * width_ + j) * image.channels + k] = image.data[((i - top_pad) * image.width + j - left_pad) * image.channels + k];
                        }
                    }
                }
            }
            cropped_images.emplace_back(cropped_image, width_, height_, image.channels, image.original_width, image.original_height);
        }

        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        // delete &images;
        images.clear();
    }
    return cropped_images;
}

std::vector<ImageInfo> PreProcessor::NormalizeImages(std::vector<ImageInfo> &images, vector<float> means, vector<float> stds, bool free_source) {
    auto normalized_images = std::vector<ImageInfo>();
    for (auto image : images) {
        auto normalized_image = new float[image.width * image.height * image.channels];
        // for (int i = 0; i < image.width * image.height * image.channels; i++) {
        //     normalized_image[i] = (image.data[i] - mean) / std;
        // }
        auto height = image.height;
        auto width = image.width;
        auto channel = image.channels;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < channel; k++) {
                    // std::cout << "Pixel at (" << i << ", " << j << ", " << k << "): " <<  image.data[(i * width + j) * channel + k] << std::endl;
                    auto vv = image.data[(i * width + j) * channel + k];
                    normalized_image[(i * width + j) * channel + k] = (vv - means[k]) / (stds[k]);
                }
            }
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
