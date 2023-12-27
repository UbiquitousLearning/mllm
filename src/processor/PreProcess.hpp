//
// Created by 咸的鱼 on 2023/12/21.
//

#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP
#include "tokenizers/Tokenizer.hpp"

namespace mllm {
enum PaddingType {
    CONSTANT,
};

enum ResampleType {
    BILINEAR,
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

    // Convert from CWH to WHC
    size_t convert_index(size_t cwh_idx) const {
        size_t c = cwh_idx / (width * height);
        size_t wh = cwh_idx % (width * height);
        size_t w = wh % width;
        size_t h = wh / width;
        return h * width * channels + w * channels + c;
    }

    float get_whc_pixel(size_t idx) const {
        return data[convert_index(idx)];
    }
};

class PreProcessor {
public:
    PreProcessor(Tokenizer *tokenizer) :
        tokenizer_(tokenizer) {
    }

    virtual void PreProcessImages(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length, int height = 1080, int width = 1920, bool do_pad = true, bool do_resize = true, bool do_normalize = true, float mean = 0.5, float std = 0.5) =0;
    virtual void Process(const std::string &text) =0;
    virtual void PreProcessImages(const std::vector<std::string> &images_path, int height = 1080, int width = 1920, bool do_pad = true, bool do_resize = true, bool do_normalize = true, float mean = 0.5, float std = 0.5) =0;

protected:
    Tokenizer *tokenizer_;

public:
    static float *RescaleImage(const uint8_t *data, float scale, unsigned int length);
    static std::vector<ImageInfo> PadImages(std::vector<ImageInfo> &images, int height, int width, size_t patch_width = 30, size_t patch_height = 30, float pad_value = 1.0 / 255.0, PaddingType padding_type = PaddingType::CONSTANT, bool free_source = true);
    static std::vector<ImageInfo> ResizeImages(std::vector<ImageInfo> &images, int height, int width, bool strict_size = false, ResampleType resample_type = ResampleType::BILINEAR, bool free_source = true);
    static std::vector<ImageInfo> NormalizeImages(std::vector<ImageInfo> &images, float mean, float std, bool free_source = true);
    static std::vector<ImageInfo> CenterCropImages(std::vector<ImageInfo> &images, int height, int width,float pad = 0.0F, bool free_source = true);

};
}

#endif //PREPROCESS_HPP
