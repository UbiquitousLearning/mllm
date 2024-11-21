//
// Created by Xiang Li on 2023/12/23.
//
#include "PreProcess.hpp"
#include <omp.h>

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

std::vector<ImageInfo> PreProcessor::ResizeImages(std::vector<ImageInfo> &images, int height, int width, bool strict_size, bool fit, ResizeFitEdge fit_edge, ResampleType resample_type, bool free_source) {
    // assert(resample_type == ResampleType::BILINEAR);
    stbir_filter filter = stbir_filter::STBIR_FILTER_DEFAULT;
    switch (resample_type) {
    case ResampleType::BILINEAR:
        filter = stbir_filter::STBIR_FILTER_TRIANGLE;
        break;
    case ResampleType::BICUBIC:
        filter = stbir_filter::STBIR_FILTER_CUBICBSPLINE;
        break;
    default:
        MLLM_LOG_ERROR_STREAM << "Not implemented Reshape Filter yet" << std::endl;
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

float cubicWeight(float t, float a = -0.5f) {
    t = std::abs(t);
    if (t <= 1.0f) {
        return (a + 2.0f) * t * t * t - (a + 3.0f) * t * t + 1.0f;
    } else if (t < 2.0f) {
        return a * t * t * t - 5.0f * a * t * t + 8.0f * a * t - 4.0f * a;
    }
    return 0.0f;
}
ImageInfo PreProcessor::ImageInterpolation(ImageInfo &image, int new_height, int new_width, ResampleType mode, bool free_source) {
    auto scaled_data = new float[new_width * new_height * image.channels];
    float heightScale = static_cast<float>(image.height) / new_height;
    float widthScale = static_cast<float>(image.width) / new_width;

    switch (mode) {
    case ResampleType::BICUBIC: {
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int c = 0; c < image.channels; ++c) {
            for (int y = 0; y < new_height; ++y) {
                for (int x = 0; x < new_width; ++x) {
                    float srcY = y * heightScale;
                    float srcX = x * widthScale;
                    int y0 = static_cast<int>(std::floor(srcY));
                    int x0 = static_cast<int>(std::floor(srcX));
                    float dy = srcY - y0;
                    float dx = srcX - x0;
                    float result = 0.0f;
                    for (int j = -1; j <= 2; ++j) {
                        for (int i = -1; i <= 2; ++i) {
                            int sampleY = std::clamp(y0 + j, 0, image.height - 1);
                            int sampleX = std::clamp(x0 + i, 0, image.width - 1);
                            float pixelValue = image.data[(sampleY * image.width + sampleX) * image.channels + c];
                            float weight = cubicWeight(j - dy) * cubicWeight(i - dx);
                            result += pixelValue * weight;
                        }
                    }
                    scaled_data[(y * new_width + x) * image.channels + c] = result;
                }
            }
        }
        break;
    }
    case ResampleType::BILINEAR: {
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int c = 0; c < image.channels; ++c) {
            for (int y = 0; y < new_height; ++y) {
                for (int x = 0; x < new_width; ++x) {
                    float srcY = y * heightScale;
                    float srcX = x * widthScale;
                    int y0 = static_cast<int>(srcY);
                    int x0 = static_cast<int>(srcX);
                    int y1 = std::min(y0 + 1, image.height - 1);
                    int x1 = std::min(x0 + 1, image.width - 1);
                    float ly = srcY - y0;
                    float lx = srcX - x0;
                    float hy = 1.0f - ly;
                    float hx = 1.0f - lx;
                    float v0 = hx * hy * image.data[(y0 * image.width + x0) * image.channels + c];
                    float v1 = lx * hy * image.data[(y0 * image.width + x1) * image.channels + c];
                    float v2 = hx * ly * image.data[(y1 * image.width + x0) * image.channels + c];
                    float v3 = lx * ly * image.data[(y1 * image.width + x1) * image.channels + c];
                    scaled_data[(y * new_width + x) * image.channels + c] = v0 + v1 + v2 + v3;
                }
            }
        }
        break;
    }
    case ResampleType::DEFAULT:
    default: {
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
        for (int i = 0; i < image.height * image.width * image.channels; ++i) {
            for (int c = 0; c < image.channels; ++c) {
                for (int y = 0; y < new_height; ++y) {
                    for (int x = 0; x < new_width; ++x) {
                        int originalY = static_cast<int>(y * heightScale);
                        int originalX = static_cast<int>(x * widthScale);
                        scaled_data[(y * new_width + x) * image.channels + c] =
                            image.data[(originalY * image.width + originalX) * image.channels + c];
                    }
                }
            }
        }
    }
    }
    auto scaledImageInfo = ImageInfo(scaled_data, new_width, new_height, image.channels);
    if (free_source) {
        free(image.data);
        image.data = nullptr;
    }
    return scaledImageInfo;
}

ImageInfo PreProcessor::ImageTranspose(ImageInfo &image, bool free_source) {
    int new_height = image.width;
    int new_width = image.height;
    auto scaled_data = new float[new_width * new_height * image.channels];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
    for (int c = 0; c < image.channels; ++c) {
        for (int y = 0; y < new_height; ++y) {
            for (int x = 0; x < new_width; ++x) {
                scaled_data[(y * new_width + x) * image.channels + c] =
                    image.data[(x * image.width + y) * image.channels + c];
            }
        }
    }

    auto scaledImageInfo = ImageInfo(scaled_data, new_width, new_height, image.channels);
    if (free_source) {
        delete[] image.data;
        image.data = nullptr;
    }
    return scaledImageInfo;
}

void PreProcessor::ImageInfos2Pixels(std::vector<ImageInfo> &imageinfos, vector<vector<vector<vector<float>>>> &pixel_values_) {
    for (auto &imageinfo : imageinfos) {
        auto pixel_values = vector<vector<vector<float>>>(imageinfo.channels, vector<vector<float>>(imageinfo.height, vector<float>(imageinfo.width)));
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int k = 0; k < imageinfo.channels; ++k) {
            for (int i = 0; i < imageinfo.height; ++i) {
                for (int j = 0; j < imageinfo.width; ++j) {
                    pixel_values[k][i][j] = imageinfo.get_whc_pixel(i * imageinfo.width + j + k * imageinfo.width * imageinfo.height);
                }
            }
        }
#pragma omp critical
        {
            pixel_values_.push_back(pixel_values);
        }
    }
}