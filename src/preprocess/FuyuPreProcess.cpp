//
// Created by 咸的鱼 on 2023/12/4.
//

#include "FuyuPreProcess.hpp"

#include <iostream>

namespace mllm {
void FuyuPreProcess::PreProcessImages(const std::vector<vector<uint8_t>> &images, int height, int width, bool do_pad, bool do_resize, bool do_normalize, float mean, float std) {
    assert(height > 0 && width > 0);

    // if (do_resize) {
    //     // Not implemented yet
    //     std::cerr << "Resize not implemented yet" << std::endl;
    //     exit(-1);
    // }
    // auto images_ = std::vector<ImageInfo>();
    images_.resize(images.size());
    for (auto image : images) {
        int width_, height_, channels_;
        // Data is [height * width * channels],RGB
        const unsigned char *data = stbi_load_from_memory(image.data(), image.size(), &width, &height, &channels_, 3);
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
    if (do_resize) {
        images_ = ResizeImages(images_, height, width);
    }

    // TODO: PAD images
    if (do_pad) {
        images_ = PadImages(images_, height, width);
    }
    if (do_normalize) {
        images_ = NormalizeImages(images_, mean, std);
    }
    // TODO: Patch images
}

void FuyuPreProcess::Process(std::string text) {
    if (text.empty()) {
        return;
    }
    if (images_.empty()) {
        std::cout << "images is empty" << std::endl;
    }
    auto batch_size = images_.size();
}

std::vector<ImageInfo> FuyuPreProcess::PadImages(const std::vector<ImageInfo> &images, int height, int width, float pad_value, PaddingType padding_type, bool free_source) {
    assert(padding_type == PaddingType::CONSTANT);
    auto padded_images = std::vector<ImageInfo>();
    for (auto image : images) {
        if (image.height == height && image.width == width && image.channels == 3) {
            padded_images.emplace_back(image.data, image.width, image.height, image.channels, image.original_width, image.original_height);

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
        padded_images.emplace_back(padded_image.data(), width, height, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        delete &images;
    }
    return padded_images;
}

std::vector<ImageInfo> FuyuPreProcess::ResizeImages(const std::vector<ImageInfo> &images, int height, int width, ResampleType resample_type, bool free_source) {
    assert(resample_type == ResampleType::BILINEAR);
    assert(free_source=true);
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
        auto resized_image = std::vector<float>();
        resized_image.resize(height * width * image.channels);
        stbir_resize(image.data, image.width, image.height, 0, resized_image.data(), width, height, 0, stbir_pixel_layout::STBIR_RGB, stbir_datatype::STBIR_TYPE_FLOAT, stbir_edge::STBIR_EDGE_CLAMP, filter);
        resized_images.emplace_back(resized_image.data(), width, height, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        delete &images;
    }

    return resized_images;
}

vector<ImageInfo> FuyuPreProcess::NormalizeImages(const vector<ImageInfo> &images, float mean, float std, bool free_source) {
    auto normalized_images = std::vector<ImageInfo>();
    for (auto image : images) {
        auto normalized_image = std::vector<float>();
        normalized_image.resize(image.width * image.height * image.channels);
        for (int i = 0; i < image.width * image.height * image.channels; i++) {
            normalized_image[i] = (image.data[i] - mean) / std;
        }
        normalized_images.emplace_back(normalized_image.data(), image.width, image.height, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        delete &images;
    }
    return normalized_images;
}

void FuyuPreProcess::get_sample_encoding(const std::string &text) {
    for (auto &image : images_) {
        auto height = image.height;
        auto width = image.width;
        auto num_patches_per_dim_h = height / patch_size_.first;
        auto num_patches_per_dim_w = width / patch_size_.second;
        auto num_patches = num_patches_per_dim_h * num_patches_per_dim_w;
        auto tensor_of_image_ids = vector<vector<token_id_t>>(num_patches_per_dim_h, vector<token_id_t>(num_patches_per_dim_w, image_placeholder_id_));
        for (auto &row : tensor_of_image_ids) {
            row.push_back(image_newline_id_);
        }

    }

}

std::vector<vector<vector<float>>> FuyuPreProcess::PatchImages( std::vector<ImageInfo> &images, size_t patch_height, size_t patch_width) {
    auto batch = images.size();
    if (batch == 0) {
        return {};
    }
    auto image_0 = images[0];
    auto height = image_0.height;
    auto width = image_0.width;
    auto channels = image_0.channels;
    auto square = width * height;
    auto dim2 = square / patch_height / patch_width;
    auto dim_2_1 = width / patch_width;
    auto dim2_2 = height / patch_height;
    auto stride2 = patch_height * width;
    auto stride1 = patch_width;
    auto patches = vector<vector<vector<float>>>(batch, vector<vector<float>>(dim2, vector<float>()));
    for (int b = 0; b < batch; b++) {
        auto patch_ = patches[b];
        for (int i = 0; i < dim2_2; i++) {
            for (int j = 0; j < dim_2_1; j++) {
                auto patch = patch_[i * dim_2_1 + j];
                auto const index_first_element_of_line = i * stride2 + j * stride1;
                while (patch.size() < patch_height * patch_width * channels) {
                    for (int h = 0; h < patch_height; h++) {
                        for (int w = 0; w < patch_width; w++) {
                            for (int c = 0; c < channels; c++) {
                                patch.push_back(images[b].data[index_first_element_of_line + h * patch_height + w + c * square]);
                            }
                        }
                    }
                }
            }
        }
        // if (free_source) {
        //     free(images[b].data);
        //     images[b].data = nullptr;
        // }
    }
    return patches;
}
} // mllm
