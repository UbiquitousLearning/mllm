//
// Created by Rongjie Yi on 2024/2/19 0004.
//

#ifndef TOKENIZATION_CLIP_HPP
#define TOKENIZATION_CLIP_HPP

#include <utility>
#include "processor/PreProcess.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include <cstdlib>
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb/stb_image.h"

#include <numeric>

using namespace mllm;

class ClipProcessor : public PreProcessor {
public:
    BPETokenizer *tokenizer;
    vector<vector<token_id_t>> input_ids_;
    vector<vector<vector<vector<float>>>> pixel_values_;

    Tensor img2Tensor(vector<vector<vector<float>>> img, string name = "input", BackendType type = MLLM_CPU) {
        int channel = img.size();
        int height = img[0].size();
        int width = img[0][0].size();
        Tensor tensor1(1, height, channel, width, Backend::global_backends[type], true);
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int h = 0; h < height; ++h) {
            for (int c = 0; c < channel; ++c) {
                for (int w = 0; w < width; ++w) {
                    tensor1.setDataAt<float>(0, h, c, w, img[c][h][w]);
                }
            }
        }
        return tensor1;
    }
    vector<float> softmax(const vector<float> &scores) {
        vector<float> exps;
        float max_val = *max_element(scores.begin(), scores.end());
        for (float score : scores) {
            exps.push_back(exp(score - max_val));
        }
        float sum_exps = accumulate(exps.begin(), exps.end(), 0.0f);
        for (float &exp : exps) {
            exp /= sum_exps;
        }
        return exps;
    }

public:
    explicit ClipProcessor(const string &vocab_path, const string &merges_path, int height = 224, int width = 224, bool add_special_tokens = true) :
        PreProcessor(height, width, false, true, true,
                     true, {0.48145466, 0.4578275, 0.40821073}, {0.26862954, 0.26130258, 0.27577711}) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_path);
        std::unordered_map<string, unsigned> merge_rank;
        auto merge_file = std::ifstream(merges_path);
        std::string line;
        unsigned rank = 0;
        while (std::getline(merge_file, line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] == '#') {
                continue;
            }
            merge_rank[line] = rank;
            rank++;
        }
        tokenizer->setMergeRank(merge_rank);
        if (add_special_tokens) {
            tokenizer->setSpecialToken("<|startoftext|>", "<|endoftext|>");
        }
    }

    void Process(const std::string &text) override {
        auto token_id = vector<token_id_t>();
        tokenizer_->tokenize(text, token_id, false);
        input_ids_.push_back(token_id);
    }

    void PreProcessImages(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length) override {
        auto imageinfos = vector<ImageInfo>();
        for (int i = 0; i < images.size(); i++) {
            int width, height, channels;
            auto data = stbi_load_from_memory(images[i], image_length[i], &width, &height, &channels, 3);
            if (data == nullptr) {
                MLLM_LOG_ERROR_STREAM << "Error: Failed to load image from memory." << std::endl;
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
            imageinfos = PreProcessor::ResizeImages(imageinfos, height_, width_, false, true, shortest);
        }
        // std::cout<<imageinfos[0].height<< imageinfos[0].width <<std::endl;
        // Use height_ or crop_size?
        imageinfos = PreProcessor::CenterCropImages(imageinfos, height_, width_, 0, true);

        if (do_normalize_) {
            imageinfos = PreProcessor::NormalizeImages(imageinfos, mean_, std_);
        }
        // todo: Optimize this!
        for (auto &imageinfo : imageinfos) {
            auto pixel_values = vector<vector<vector<float>>>();
            for (int k = 0; k < imageinfo.channels; k++) {
                auto channel = vector<vector<float>>();
                for (int i = 0; i < imageinfo.height; i++) {
                    auto row = vector<float>();
                    for (int j = 0; j < imageinfo.width; j++) {
                        row.push_back(imageinfo.get_whc_pixel(i * imageinfo.width + j + k * imageinfo.width * imageinfo.height));
                    }
                    channel.push_back(row);
                }
                pixel_values.push_back(channel);
            }

            pixel_values_.push_back(pixel_values);
        }
    }

    void PreProcessImages(const std::vector<std::string> &images_path) override {
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
    void PreProcessImages(const std::vector<std::string> &images_path, int height, int width) {
        height_ = height;
        width_ = width;
        PreProcessImages(images_path);
    }

    vector<Tensor> process(vector<string> in_strs, string img_path, int hw = 224,
                           string img_name = "input_vision", string text_name = "input_text", BackendType type = MLLM_CPU) {
        input_ids_.clear();
        pixel_values_.clear();
        auto tokens_ids = vector<vector<token_id_t>>();
        for (auto in_str : in_strs) {
            vector<mllm::token_id_t> tokens_id = {};
            tokenizer->tokenize(in_str, tokens_id, true, true, "</w>");
            tokens_ids.push_back(tokens_id);
        }
        PreProcessImages({std::move(img_path)}, hw, hw);
        auto images = pixel_values_[0];

        return {Tokenizer::tokens2Input(tokens_ids), img2Tensor(images, std::move(img_name))};
    }
    vector<float> postProcess(Tensor &result) {
        vector<float> scores;
        for (int i = 0; i < result.batch(); ++i) {
            auto value = result.dataAt<float>(i, 0, 0, 0);
            scores.push_back(value);
        }
        auto token_idx = softmax(scores);
        return token_idx;
    }
};

#endif // TOKENIZATION_CLIP_HPP
