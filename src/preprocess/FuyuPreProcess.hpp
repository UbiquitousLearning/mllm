//
// Created by 咸的鱼 on 2023/12/4.
//

#ifndef FUYUPREPROCESS_HPP
#define FUYUPREPROCESS_HPP
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <imageHelper/stb_image.h>
#include <imageHelper/stb_image_resize2.h>

#include "tokenizers/Tokenizer.hpp"
// #include <imageHelper/stb_image_resize.h>
using std::vector;

namespace mllm {
typedef vector<vector<vector<vector<float>>>> FourDVector;

struct FuyuBatchEncoding {
    std::vector<int> input_ids;
    std::vector<int> attention_mask;
    std::vector<FourDVector> image_patches;
    std::vector<int> image_patches_indices;
};

struct ImageInfo {
    float *data;
    int width;
    int height;
    int channels;
    int original_width;
    int original_height;

    ImageInfo(float *data, int width, int height, int channels) :
        data(data), width(width), height(height), original_height(height), original_width(width), channels(channels) {
    }

    ImageInfo(float *data, int width, int height, int channels, int original_width, int original_height
        ) :
        data(data), width(width), height(height), channels(channels), original_width(original_width), original_height(original_height) {
    }
};

enum PaddingType {
    CONSTANT,
};

enum ResampleType {
    BILINEAR,
};

class FuyuPreProcess {
    std::vector<ImageInfo> images_;
    Tokenizer *tokenizer_;
    token_id_t image_placeholder_id_;
    token_id_t image_newline_id_;
    std::pair<size_t,size_t> patch_size_;

public:
    explicit FuyuPreProcess(Tokenizer *tokenizer,size_t patch_height = 30, size_t patch_width = 30) :
        tokenizer_(tokenizer) {
        auto tmp_token = vector<token_id_t>();
        tokenizer_->tokenize("|SPEAKER|", tmp_token, false);
        image_placeholder_id_ = tmp_token[0];
        tokenizer_->tokenize("|NEWLINE|", tmp_token, false);
        image_newline_id_ = tmp_token[0];
        patch_size_ = std::make_pair(patch_height, patch_width);
    }

    void PreProcessImages(const std::vector<std::vector<uint8_t>> &images, int height = 224, int width = 224, bool do_pad = false, bool do_resize = false, bool do_normalize = false, float mean = 0.5, float std = 0.5);
    void Process(std::string text);

private:
    static std::vector<ImageInfo> PadImages(const std::vector<ImageInfo> &images, int height, int width, float pad_value = 1.0, PaddingType padding_type = PaddingType::CONSTANT, bool free_source = true);
    static std::vector<ImageInfo> ResizeImages(const std::vector<ImageInfo> &images, int height, int width, ResampleType resample_type = ResampleType::BILINEAR, bool free_source = true);
    static vector<ImageInfo> NormalizeImages(const vector<ImageInfo> &images, float mean, float std, bool free_source = true);
    void get_sample_encoding(const std::string &text);
    static std::vector<vector<vector<float>>> PatchImages( std::vector<ImageInfo> &images, size_t patch_height, size_t patch_width);


};
} // mllm

#endif //FUYUPREPROCESS_HPP
