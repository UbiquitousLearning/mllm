//
// Created by Rongjie Yi on 2024/2/19 0004.
//

#ifndef TOKENIZATION_VIT_HPP
#define TOKENIZATION_VIT_HPP

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "processor/PreProcess.hpp"
#include "stb/stb_image.h"

using namespace mllm;

class ViTProcessor final : public PreProcessor {
    Tensor img2Tensor(float *img, int height, int width, int channel, string name = "input", BackendType type = MLLM_CPU) {
        Tensor tensor1(1, height, channel, width, Backend::global_backends[type].get(), true);
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int h = 0; h < height; ++h) {
            for (int c = 0; c < channel; ++c) {
                for (int w = 0; w < width; ++w) {
                    tensor1.setDataAt<float>(0, h, c, w, img[(h * width + w) * channel + c]);
                }
            }
        }
        return tensor1;
    }
    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        unsigned int maxIndex = 0;
        float maxValue = scores[0];
        for (size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > maxValue) {
                maxIndex = i;
                maxValue = scores[i];
            }
        }
        return maxIndex;
    }

public:
    explicit ViTProcessor() :
        PreProcessor(224, 224, false, true, true,
                     true, {0.5}, {0.5}) {
        Module::initBackend(MLLM_CPU);
    }

    Tensor process(string img_path, int hw = 224, string name = "input", BackendType type = MLLM_CPU) {
        int width_, height_, channel_;
        unsigned char *data = stbi_load(img_path.c_str(), &width_, &height_, &channel_, 0); //"../assets/cat.jpg"
        if (data == nullptr) {
            std::cout << "load image failed" << std::endl;
        }
        auto data_f32 = RescaleImage(data, 255.0, height_ * width_ * channel_);
        auto images = std::vector<ImageInfo>({ImageInfo(data_f32, width_, height_, channel_)});
        images = ResizeImages(images, hw, hw, true);
        images = NormalizeImages(images, 0.5, 0.5);
        data_f32 = images[0].data;
        stbi_image_free(data);
        auto input_tensor = img2Tensor(data_f32, hw, hw, channel_);
        return input_tensor;
    }
    unsigned int postProcess(Tensor &result) {
        assert(result.batch() == 1);
        assert(result.head() == 1);
        vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, 0, i);
            scores.push_back(value);
        }
        auto token_idx = argmax(scores);
        return token_idx;
    }

    void PreProcessImages(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length) override {};
    void Process(const std::string &text) override {};
};

#endif // TOKENIZATION_VIT_HPP
