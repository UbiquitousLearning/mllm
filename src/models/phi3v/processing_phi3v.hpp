//
// Created by Rongjie Yi on 24-3-8.
//

#ifndef PROCESSING_Phi3V_HPP
#define PROCESSING_Phi3V_HPP
#include <iostream>
#include "OpDefined.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "processor/PreProcess.hpp"
#include "tokenizers/Tokenizer.hpp"
#include "models/phi3/tokenization_phi3.hpp"
#include <cassert>
#include <cstddef>
#include <utility>
#include <regex>
#include <vector>
#include <cstdlib>
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb/stb_image.h"

using namespace mllm;

struct Phi3VImageDatas {
    Tensor pixel_values;
    vector<std::pair<size_t, size_t>> image_sizes;
    vector<int> num_img_tokens;
};
class Phi3VImageProcessor {
    int num_crops = 16;
    std::vector<float> mean_ = {0.48145466, 0.4578275, 0.40821073};
    std::vector<float> std_ = {0.26862954, 0.26130258, 0.27577711};
    ImageInfo padding_336(ImageInfo &image, float pad_data, bool free_source = true) {
        // auto cropped_images = std::vector<ImageInfo>();
        auto height = image.height;
        auto width = image.width;
        int new_height, new_width;
        int top_pad, bottom_pad, left_pad, right_pad;
        if (width >= height) {
            new_height = static_cast<int>(std::ceil(static_cast<double>(height) / 336) * 336);
            left_pad = 0;
            right_pad = width;
            top_pad = int((new_height - height) / 2);
            bottom_pad = height + top_pad; // tar - (tar - height - top_pad);
            new_width = width;
        } else { // width < height
            new_width = static_cast<int>(std::ceil(static_cast<double>(width) / 336) * 336);
            top_pad = 0;
            bottom_pad = height;
            left_pad = int((new_width - width) / 2);
            right_pad = width + left_pad; // tar - (tar - width - left_pad);
            new_height = height;
        }
        auto cropped_image = new float[new_height * new_width * image.channels];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int i = 0; i < new_height; i++) {
            for (int j = 0; j < new_width; j++) {
                for (int k = 0; k < image.channels; k++) {
                    if (i < top_pad || i >= bottom_pad || j < left_pad || j >= right_pad) {
                        cropped_image[(i * new_width + j) * image.channels + k] = pad_data;
                    } else {
                        cropped_image[(i * new_width + j) * image.channels + k] = image.data[((i - top_pad) * image.width + j - left_pad) * image.channels + k];
                    }
                }
            }
        }
        auto image_info = ImageInfo(cropped_image, new_width, new_height, image.channels);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
        return image_info;
    }

public:
    vector<int> num_img_tokens;
    vector<std::pair<size_t, size_t>> image_sizes;
    vector<vector<vector<vector<float>>>> pixel_values;
    vector<vector<vector<vector<float>>>> global_pixel_values;

public:
    explicit Phi3VImageProcessor() {
    }
    vector<vector<token_id_t>> input_ids_;
    void preprocess_images(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length) {
        assert(images.size() == 1);
        auto imageinfos = vector<ImageInfo>();
        for (int i = 0; i < images.size(); i++) {
            int width, height, channels;
            auto data = stbi_load_from_memory(images[i], image_length[i], &width, &height, &channels, 3);
            if (data == nullptr) {
                MLLM_LOG_ERROR_STREAM << "Error: Failed to load image from memory." << std::endl;
                exit(-1);
            }
            float *f32_data = nullptr;
            f32_data = PreProcessor::RescaleImage(data, 225, width * height * channels);
            stbi_image_free(data);
            auto image_info = ImageInfo(f32_data, width, height, channels);
            // HD_transform
            int hd_num = num_crops;
            height = image_info.height;
            width = image_info.width;
            int new_w = width;
            int new_h = height;
            bool trans = false;
            if (width < height) {
                trans = true;
                image_info = PreProcessor::ImageTranspose(image_info);
            }
            float ratio = float(width) / float(height);
            auto scale = 1;
            while (scale * std::ceil(scale / ratio) <= hd_num) {
                scale += 1;
            }
            scale -= 1;
            new_w = int(scale * 336);
            new_h = int(new_w / ratio);
            std::vector<ImageInfo> temp_image_info = {image_info};
            temp_image_info = PreProcessor::ResizeImages(temp_image_info, new_h, new_w, true, false, ResizeFitEdge::shortest, BILINEAR, true);
            image_info = padding_336(temp_image_info[0], 1, true);
            if (trans) {
                image_info = PreProcessor::ImageTranspose(image_info);
            }
            imageinfos.emplace_back(image_info);
        }
        imageinfos = PreProcessor::NormalizeImages(imageinfos, mean_, std_);

        for (auto &image_info : imageinfos) {
            auto h = image_info.height;
            auto w = image_info.width;
            image_sizes.push_back({h, w});
            int num_img_token = int(((h / 336) * (w / 336) + 1) * 144 + 1 + (h / 336 + 1) * 12);
            num_img_tokens.push_back(num_img_token);
        }
        auto global_image = vector<ImageInfo>();
        for (int i = 0; i < imageinfos.size(); i++) {
            global_image.push_back(PreProcessor::ImageInterpolation(imageinfos[i], 336, 336, ResampleType::BICUBIC, false));
        }
        PreProcessor::ImageInfos2Pixels(imageinfos, pixel_values);
        PreProcessor::ImageInfos2Pixels(global_image, global_pixel_values);
    }

    Tensor getTensor(vector<vector<vector<vector<float>>>> imgs, vector<vector<vector<vector<float>>>> global_imgs, string name = "pixel_values", BackendType type = MLLM_CPU) {
        int batch_size = imgs.size();
        int time_all = num_crops + 1;
        for (int ii = 0; ii < batch_size; ii++) {
            auto img = imgs[ii];
            auto global_img = global_imgs[ii];
            int channel = img.size();
            int height = img[0].size();
            int width = img[0][0].size();
            int h_times = height / 336;
            int w_times = width / 336;
            int times = 1 + h_times * w_times;
            if (times < num_crops + 1) {
                times = num_crops + 1;
            }
            if (time_all < times) {
                time_all = times;
            }
        }
        Tensor tensor1(Backend::global_backends[type]);
        tensor1.reshape(batch_size, imgs[0].size(), time_all, 336, 336);
        tensor1.alloc();
        memset(tensor1.hostPtr<float>(), 0, tensor1.count() * sizeof(float));
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int ii = 0; ii < batch_size; ii++) {
            auto img = imgs[ii];
            auto global_img = global_imgs[ii];
            int channel = img.size();
            int height = img[0].size();
            int width = img[0][0].size();
            int h_times = height / 336;
            int w_times = width / 336;
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
            for (int h = 0; h < 336; ++h) {
                for (int w = 0; w < 336; ++w) {
                    for (int c = 0; c < channel; ++c) {
                        auto value = global_img[c][h][w];
                        tensor1.setDataAt<float>(ii, c, 0, h, w, value);
                    }
                }
            }
            // time: [1, h_times * w_times)
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
            for (int ht = 0; ht < h_times; ++ht) {
                for (int wt = 0; wt < w_times; ++wt) {
                    for (int h = 0; h < 336; ++h) {
                        for (int w = 0; w < 336; ++w) {
                            for (int c = 0; c < channel; ++c) {
                                auto value = img[c][ht * 336 + h][wt * 336 + w];
                                tensor1.setDataAt<float>(ii, c, ht * w_times + wt + 1, h, w, value);
                            }
                        }
                    }
                }
            }
        }
        return tensor1;
    }
    Tensor getTensorFlatten(vector<vector<vector<vector<float>>>> imgs, vector<vector<vector<vector<float>>>> global_imgs, string name = "pixel_values", BackendType type = MLLM_CPU) {
        int batch_size = imgs.size();
        int time_all = num_crops + 1;
        for (int ii = 0; ii < batch_size; ii++) {
            auto img = imgs[ii];
            auto global_img = global_imgs[ii];
            int channel = img.size();
            int height = img[0].size();
            int width = img[0][0].size();
            int h_times = height / 336;
            int w_times = width / 336;
            int times = 1 + h_times * w_times;
            if (times < num_crops + 1) {
                times = num_crops + 1;
            }
            if (time_all < times) {
                time_all = times;
            }
        }
        Tensor tensor1(Backend::global_backends[type]);
        tensor1.reshape(batch_size * time_all, 336, imgs[0].size(), 336);
        tensor1.alloc();
        memset(tensor1.hostPtr<float>(), 0, tensor1.count() * sizeof(float));
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        int base_batch = 0;
        for (int ii = 0; ii < batch_size; ii++) {
            auto img = imgs[ii];
            auto global_img = global_imgs[ii];
            int channel = img.size();
            int height = img[0].size();
            int width = img[0][0].size();
            int h_times = height / 336;
            int w_times = width / 336;
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
            for (int h = 0; h < 336; ++h) {
                for (int w = 0; w < 336; ++w) {
                    for (int c = 0; c < channel; ++c) {
                        auto value = global_img[c][h][w];
                        tensor1.setDataAt<float>(base_batch + 0, h, c, w, value);
                    }
                }
            }
            // time: [1, h_times * w_times)
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
            for (int ht = 0; ht < h_times; ++ht) {
                for (int wt = 0; wt < w_times; ++wt) {
                    for (int h = 0; h < 336; ++h) {
                        for (int w = 0; w < 336; ++w) {
                            for (int c = 0; c < channel; ++c) {
                                auto value = img[c][ht * 336 + h][wt * 336 + w];
                                tensor1.setDataAt<float>(base_batch + ht * w_times + wt + 1, h, c, w, value);
                            }
                        }
                    }
                }
            }
            base_batch += time_all;
        }
        return tensor1;
    }
    Phi3VImageDatas process(const std::vector<std::string> &images_path, bool flatten_img = true) {
        // assert(height_ > 0 && width_ > 0);
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
        preprocess_images(image_data, image_length);
        Tensor img_tensor;
        if (flatten_img) {
            img_tensor = getTensorFlatten(pixel_values, global_pixel_values);
        } else {
            img_tensor = getTensor(pixel_values, global_pixel_values);
        }
        return {
            img_tensor,
            image_sizes,
            num_img_tokens};
    }
};

class Phi3VProcessor final { //} : public PreProcessor {
    unsigned int argmax(const vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

    std::vector<std::string> re_split(const std::string &s, const std::string &pattern) {
        std::vector<std::string> result;
        std::regex re(pattern);
        std::sregex_token_iterator iter(s.begin(), s.end(), re, -1);
        std::sregex_token_iterator end;
        while (iter != end) {
            result.push_back(*iter++);
        }
        return result;
    }
    std::vector<std::string> re_findall(const std::string &s, const std::string &pattern) {
        std::vector<std::string> result;
        std::regex re(pattern);
        std::sregex_iterator iter(s.begin(), s.end(), re);
        std::sregex_iterator end;
        while (iter != end) {
            result.push_back(iter->str());
            ++iter;
        }
        return result;
    }
    std::vector<int> image_tags2ids(const std::vector<std::string> &image_tags) {
        std::vector<int> image_ids;
        for (const std::string &s : image_tags) {
            std::istringstream iss(s);
            std::string part;
            std::getline(iss, part, '|');
            std::getline(iss, part, '|');
            std::istringstream partIss(part);
            std::string subpart;
            while (std::getline(partIss, subpart, '_')) {}
            image_ids.push_back(std::stoi(subpart));
        }
        return image_ids;
    }

    static Tensor tokens2Input(vector<vector<int>> tokens, string name = "input", BackendType type = MLLM_CPU) {
        const auto bsize = static_cast<int>(tokens.size());
        Tensor tensor1(bsize, 1, static_cast<int>(tokens[0].size()), 1, Backend::global_backends[type], true);
        tensor1.setName(name);
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int b = 0; b < bsize; ++b) {
            for (int idx = 0; idx < tokens[b].size(); ++idx) {
                tensor1.setDataAt<float>(b, 0, idx, 0, tokens[b][idx]);
            }
        }
        return tensor1;
    }
    Tensor imgpos2Tensor(vector<std::pair<size_t, size_t>> img_pos, string name = "input_img_pos", BackendType type = MLLM_CPU) {
        int num_imgs = img_pos.size();
        Tensor tensor2(1, 1, num_imgs, 2, type, true);
        tensor2.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor2.setTtype(INPUT_TENSOR);
        for (int i = 0; i < num_imgs; ++i) {
            tensor2.setDataAt<float>(0, 0, i, 0, img_pos[i].first);
            tensor2.setDataAt<float>(0, 0, i, 1, img_pos[i].second);
        }
        return tensor2;
    }

public:
    Phi3VImageProcessor image_processor;
    Phi3Tokenizer *tokenizer;
    explicit Phi3VProcessor(const string &vocab_path, const string &merges_path = "") {
        Module::initBackend(MLLM_CPU);
        tokenizer = new Phi3Tokenizer(vocab_path);
        if (!merges_path.empty()) {
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
        }
        tokenizer->set_chat_template("<|user|>\n", "<|end|>\n<|assistant|>");
    }

    vector<Tensor> process(const string text, string img_path, bool flatten_img = true, BackendType type = MLLM_CPU) {
        string new_text = text;
        if (!img_path.empty()) {
            auto image_inputs = image_processor.process({std::move(img_path)}, flatten_img);
            auto img_tensor = image_inputs.pixel_values;
            auto num_img_tokens = image_inputs.num_img_tokens;
            auto image_sizes = image_inputs.image_sizes;

            std::string pattern = "<\\|image_\\d+\\|>";
            std::vector<std::string> result = re_split(new_text, pattern);
            vector<vector<int>> prompt_chunks;
            for (auto rtest : result) {
                auto prompt_chunk = tokenizer->tokenize_vector(rtest);
                prompt_chunks.push_back(prompt_chunk);
            }
            auto image_tags = re_findall(new_text, pattern);
            auto image_ids = image_tags2ids(image_tags);
            vector<vector<int>> image_ids_pad;
            for (int i = 0; i < image_ids.size(); i++) {
                vector<int> image_id_pad(num_img_tokens[i], -image_ids[i]);
                image_ids_pad.push_back(image_id_pad);
            }
            if (prompt_chunks.size() > image_ids_pad.size()) {
                for (int i = 0; i < prompt_chunks.size() - image_ids_pad.size(); i++) {
                    vector<int> image_id_pad(0, 0);
                    image_ids_pad.push_back(image_id_pad);
                }
            }
            vector<int> tokens_id = {};
            for (int i = 0; i < prompt_chunks.size(); i++) {
                tokens_id.insert(tokens_id.end(), prompt_chunks[i].begin(), prompt_chunks[i].end());
                tokens_id.insert(tokens_id.end(), image_ids_pad[i].begin(), image_ids_pad[i].end());
            }
            return {tokens2Input({tokens_id}, std::move("input_ids")),
                    img_tensor,
                    imgpos2Tensor(image_sizes, "image_sizes")};
        } else {
            auto input_tensor = tokenizer->tokenize(new_text);
            return {input_tensor};
        }
    }

    std::string detokenize(const vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) {
        assert(result.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(result.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        vector<float> scores;
        int _dims = result.dimension();
        int _seq = result.sequence() - 1;
        for (int i = 0; i < _dims; ++i) {
            auto value = result.dataAt<float>(0, 0, _seq, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        auto text = tokenizer->detokenize({token_idx});
        text = std::regex_replace(text, std::regex("▁"), " ");
        return make_pair(text, token_idx);
    }
};
#endif // PROCESSING_Phi3V_HPP
