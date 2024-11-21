//
// Created by Xiang Li on 2023/12/21.
//

#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP
#include <utility>

#include "tokenizers/Tokenizer.hpp"
#include "AudioProcess.hpp"

namespace mllm {
enum PaddingType {
    CONSTANT,
};

enum ResampleType {
    BILINEAR,
    BICUBIC,
    DEFAULT,
};
enum ResizeFitEdge {
    none,
    shortest,
    longest,
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

    ImageInfo(float *data, int width, int height, int channels, int original_width, int original_height) :
        data(data), width(width), height(height), channels(channels), original_width(original_width), original_height(original_height) {
    }

    // Convert from CWH to WHC
    size_t convert_index_to_whc(size_t cwh_idx) const {
        size_t c = cwh_idx / (width * height);
        size_t wh = cwh_idx % (width * height);
        size_t w = wh % width;
        size_t h = wh / width;
        return h * width * channels + w * channels + c;
    }
    size_t convert_index_to_cwh(size_t whc_idx) const {
        size_t c = whc_idx % channels;
        size_t wh = whc_idx / channels;
        size_t w = wh % width;
        size_t h = wh / width;
        return c * width * height + h * width + w;
    }

    float get_whc_pixel(size_t idx) const {
        return data[convert_index_to_whc(idx)];
    }
    float get_cwh_pixel(size_t idx) const {
        return data[convert_index_to_cwh(idx)];
    }
};

class PreProcessor {
public:
    virtual ~PreProcessor() = default;

    explicit PreProcessor(Tokenizer *tokenizer, int height, int width, bool do_pad, bool do_resize, bool do_normalize, bool do_rescale, std::vector<float> mean = {}, std::vector<float> std = {}) :
        tokenizer_(tokenizer), height_(height), width_(width), do_pad_(do_pad), do_resize_(do_resize), do_normalize_(do_normalize), do_rescale_(do_rescale), mean_(std::move(mean)), std_(std::move(std)) {
    }

    explicit PreProcessor(int height, int width, bool do_pad, bool do_resize, bool do_normalize, bool do_rescale, std::vector<float> mean = {}, std::vector<float> std = {}) :
        height_(height), width_(width), do_pad_(do_pad), do_resize_(do_resize), do_normalize_(do_normalize), do_rescale_(do_rescale), mean_(std::move(mean)), std_(std::move(std)) {
    }

    virtual void PreProcessImages(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length) = 0;
    virtual void Process(const std::string &text) = 0;
    virtual void PreProcessImages(const std::vector<std::string> &images_path);
    void PreProcessImages(const std::vector<std::string> &images_path, int height, int width) {
        height_ = height;
        width_ = width;
        PreProcessImages(images_path);
    }

protected:
    Tokenizer *tokenizer_;
    int height_ = 0;
    int width_ = 0;
    bool do_pad_ = false;
    bool do_resize_ = false;
    bool do_normalize_ = false;
    bool do_rescale_ = true;
    float scale_ = 255.0;
    std::vector<float> mean_ = {0.5};
    std::vector<float> std_ = {0.5};

public:
    static float *RescaleImage(const uint8_t *data, float scale, unsigned int length);
    static std::vector<ImageInfo> PadImages(std::vector<ImageInfo> &images, int height, int width, size_t patch_width = 30, size_t patch_height = 30, float pad_value = 1.0 / 255.0, PaddingType padding_type = PaddingType::CONSTANT, bool free_source = true);
    static std::vector<ImageInfo> ResizeImages(std::vector<ImageInfo> &images, int height, int width, bool strict_size = false, bool fit = false, ResizeFitEdge fit_edge = none, ResampleType resample_type = ResampleType::BILINEAR, bool free_source = true);
    static std::vector<ImageInfo> NormalizeImages(std::vector<ImageInfo> &images, float mean, float std, bool free_source = true);
    static std::vector<ImageInfo> NormalizeImages(std::vector<ImageInfo> &images, vector<float> means, vector<float> stds, bool free_source = true);
    static std::vector<ImageInfo> CenterCropImages(std::vector<ImageInfo> &images, int height, int width, float pad = 0.0F, bool free_source = true);
    static ImageInfo ImageInterpolation(ImageInfo &image, int new_height, int new_width, ResampleType mode = ResampleType::BICUBIC, bool free_source = true);
    static ImageInfo ImageTranspose(ImageInfo &image, bool free_source = true);
    static std::vector<ImageInfo> ImagesInterpolation(std::vector<ImageInfo> &images, int new_height, int new_width, ResampleType mode = ResampleType::BICUBIC, bool free_source = true) {
        std::vector<ImageInfo> scaled_images;
        for (auto &image : images) {
            scaled_images.push_back(ImageInterpolation(image, new_height, new_width, mode, free_source));
        }
        return scaled_images;
    }
    static void ImageInfos2Pixels(std::vector<ImageInfo> &imageinfos, vector<vector<vector<vector<float>>>> &pixel_values_);

    static std::vector<std::vector<std::vector<std::vector<float>>>> ProcessAudio(std::vector<std::string> waves) {
        return ProcessWAV(waves);
    }

    // virtual vector<Tensor> process(std::string &text, vector<string> image) = 0;
    // virtual std::pair<std::string, unsigned> detokenize(Tensor &result) = 0;

    // virtual std::pair<bool, std::string> postprocess(std::string &text) = 0;
};
} // namespace mllm

#endif // PREPROCESS_HPP
