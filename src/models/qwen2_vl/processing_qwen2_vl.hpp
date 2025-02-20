//
// Created by Rongjie Yi on 25-2-9.
//

#ifndef PROCESSING_Qwen2VL_HPP
#define PROCESSING_Qwen2VL_HPP
#include <iostream>
#include "OpDefined.hpp"
#include "processor/PreProcess.hpp"
#include "tokenizers/Tokenizer.hpp"
#include "models/qwen/tokenization_qwen.hpp"
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


#include <cmath>
#include <stdexcept>

using namespace std;
using namespace mllm;

Tensor vector3d2Tensor(vector<vector<vector<float>>> img, string name = "input", BackendType type = MLLM_CPU) {
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
Tensor vector3d2Tensor(vector<vector<vector<int>>> img, string name = "input", BackendType type = MLLM_CPU) {
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
                tensor1.setDataAt<float>(0, h, c, w, (float)img[c][h][w]);
            }
        }
    }
    return tensor1;
}

class Qwen2VLImageProcessor {
public:
    int merge_size = 2;
private:
    std::vector<float> mean_ = {0.48145466, 0.4578275, 0.40821073};
    std::vector<float> std_ = {0.26862954, 0.26130258, 0.27577711};
    int IMAGE_FACTOR = 28;
    int MIN_PIXELS = 4 * 28 * 28;
    int MAX_PIXELS = 16384 * 28 * 28;
    int MAX_RATIO = 200;
    int temporal_patch_size = 2;
    int patch_size = 14;

    void viewTensor(Tensor& tensor1) {
        assert(3* 2*14*14==tensor1.dimension());
        tensor1.reshape(tensor1.sequence(), 3, 2, 14,14);
    }


    std::pair<int, int> smart_resize(int height, int width, int factor = 28, 
        int min_pixels = 3136, int max_pixels = 12845056) {
        // Check aspect ratio condition
        if (std::max(height, width) / static_cast<float>(std::min(height, width)) > MAX_RATIO) {
        throw std::invalid_argument("Absolute aspect ratio must be smaller than " + std::to_string(MAX_RATIO));
        }

        auto round_by_factor = [](int value, int f) { return ((value + f / 2) / f) * f; };
        auto floor_by_factor = [](float value, int f) { return static_cast<int>(std::floor(value / f)) * f; };
        auto ceil_by_factor = [](float value, int f) { return static_cast<int>(std::ceil(value / f)) * f; };

        int h_bar = std::max(factor, round_by_factor(height, factor));
        int w_bar = std::max(factor, round_by_factor(width, factor));

        if (h_bar * w_bar > max_pixels) {
        float beta = std::sqrt((height * width) / static_cast<float>(max_pixels));
        h_bar = floor_by_factor(height / beta, factor);
        w_bar = floor_by_factor(width / beta, factor);
        } else if (h_bar * w_bar < min_pixels) {
        float beta = std::sqrt(min_pixels / static_cast<float>(height * width));
        h_bar = ceil_by_factor(height * beta, factor);
        w_bar = ceil_by_factor(width * beta, factor);
        }
        return {h_bar, w_bar};
    }

    ImageInfo fetch_image(ImageInfo &image){
        int old_height = image.height;
        int old_width = image.width;
        auto [new_height, new_width] = smart_resize(old_height, old_width, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS);
        std::vector<ImageInfo> temp_image_info = {image};
        auto image_n = PreProcessor::ResizeImages({temp_image_info}, new_height, new_width, true)[0];
        // delete[] image.data;
        // image.data = nullptr;
        return image_n;
    }

    pair<vector<vector<float>>, vector<int>> convertPatches(
        const vector<vector<vector<vector<float>>>>& imgs,
        int temporal_patch_size,
        int patch_size,
        int merge_size,
        int resized_height,
        int resized_width) {
        int batch = imgs.size();
        int channel = (batch == 0) ? 0 : imgs[0].size();
        vector<int> shape = {0, 0, 0};
        // 检查输入有效性
        if (batch == 0 || channel == 0 || 
            batch % temporal_patch_size != 0 ||
            resized_height % patch_size != 0 ||
            resized_width % patch_size != 0 ||
            (resized_height / patch_size) % merge_size != 0 ||
            (resized_width / patch_size) % merge_size != 0) {
            return make_pair(vector<vector<float>>(), shape);
        }
        // 计算网格维度
        int grid_t = batch / temporal_patch_size;
        int grid_h = resized_height / patch_size;
        int grid_w = resized_width / patch_size;
        shape = {grid_t, grid_h, grid_w};
        // 计算最终矩阵维度
        int rows = grid_t * grid_h * grid_w;
        int cols = channel * temporal_patch_size * patch_size * patch_size;
        vector<vector<float>> flatten_patches(rows, vector<float>(cols, 0.0f));
        // 预处理常用值
        int ghm = grid_h / merge_size;
        int gwm = grid_w / merge_size;
        int ms = merge_size;
        int area_per_row = ghm * gwm * ms * ms;
        // 遍历所有输出元素
        for (int i = 0; i < rows; ++i) {
            // 计算时空块坐标
            int d0 = i / area_per_row;
            int remaining = i % area_per_row;
            int d1 = remaining / (gwm * ms * ms);
            remaining %= (gwm * ms * ms);
            int d2 = remaining / (ms * ms);
            remaining %= (ms * ms);
            int d3 = remaining / ms;
            int d4 = remaining % ms;
            for (int j = 0; j < cols; ++j) {
                // 解析通道和时间信息
                int d5 = j / (temporal_patch_size * patch_size * patch_size);
                int remaining_j = j % (temporal_patch_size * patch_size * patch_size);
                int d6 = remaining_j / (patch_size * patch_size);
                remaining_j %= (patch_size * patch_size);
                int d7 = remaining_j / patch_size;
                int d8 = remaining_j % patch_size;
                // 计算原始坐标
                int b = d0 * temporal_patch_size + d6;
                int c = d5;
                int h = ((d1 * ms + d3) * patch_size) + d7;
                int w = ((d2 * ms + d4) * patch_size) + d8;
                // 边界检查并赋值
                if (b < batch && c < channel && 
                    h < resized_height && w < resized_width &&
                    imgs[b].size() > c &&
                    imgs[b][c].size() > h &&
                    imgs[b][c][h].size() > w) {
                    flatten_patches[i][j] = imgs[b][c][h][w];
                }
            }
        }
    
        return make_pair(flatten_patches, shape);
    }

public:
    explicit Qwen2VLImageProcessor() {
    }
    vector<vector<token_id_t>> input_ids_;
    pair<vector<vector<float>>, vector<int>> preprocess_images(const uint8_t * image, const size_t &image_length) {
        auto imageinfos = vector<ImageInfo>();
        int width, height, channels;
        auto data = stbi_load_from_memory(image, image_length, &width, &height, &channels, 3);
        if (data == nullptr) {
            MLLM_LOG_ERROR_STREAM << "Error: Failed to load image from memory." << std::endl;
            exit(-1);
        }
        float *f32_data = nullptr;
        f32_data = PreProcessor::RescaleImage(data, 255, width * height * channels);
        stbi_image_free(data);
        auto image_info = ImageInfo(f32_data, width, height, channels);
        image_info = fetch_image(image_info);
        imageinfos.emplace_back(image_info);
        imageinfos = PreProcessor::NormalizeImages(imageinfos, mean_, std_);
        imageinfos.emplace_back(imageinfos[0]);
        vector<vector<vector<vector<float>>>> pixel_v;
        PreProcessor::ImageInfos2Pixels(imageinfos, pixel_v);
        auto result_patches= convertPatches(pixel_v, 
            temporal_patch_size,
            patch_size,
            merge_size,
            imageinfos[0].height,  // resized_height
            imageinfos[0].width  // resized_width
        );
        return result_patches;
    }

    pair<Tensor, vector<vector<int>>> process(const std::vector<std::string> &images_path, bool view_img = true) {
        vector<vector<vector<float>>> pixel_values;
        vector<vector<int>> vision_grid_thws;
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
            auto result_patches = preprocess_images(data, size);
            auto flatten_patches = result_patches.first;
            auto grid_thw= result_patches.second;
            pixel_values.push_back(flatten_patches);
            vision_grid_thws.push_back(grid_thw);
        }
        auto pixel_values_tensor = vector3d2Tensor(pixel_values,"pixel_values");
        if(view_img){
            assert(3* 2*14*14==pixel_values_tensor.dimension());
            pixel_values_tensor.reshape(pixel_values_tensor.head(), 3, 2, 14,14);
        }
        return {pixel_values_tensor, vision_grid_thws};
    }
};

class Qwen2VLProcessor final { //} : public PreProcessor {
    unsigned int argmax(const vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }
    // 预定义需要替换的标记
    const string IMAGE_PAD = "<|image_pad|>";
    const string PLACEHOLDER = "<|placeholder|>";

public:
    Qwen2VLImageProcessor image_processor;
    QWenTokenizer *tokenizer;
    explicit Qwen2VLProcessor(const string &vocab_path, const string &merge_path = "") {
        Module::initBackend(MLLM_CPU);
        tokenizer = new QWenTokenizer(vocab_path, merge_path);
        tokenizer->special_tokens = {
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|video_pad|>",
        };
        tokenizer->setSpecialTokenMap({{"<|image_pad|>", 151655}, {"<|video_pad|>", 151656}});
        string system_prompt_start = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
        string system_prompt_end = "<|im_end|>\n<|im_start|>assistant\n";
        tokenizer->set_chat_template(system_prompt_start, system_prompt_end);
    }

    vector<Tensor> process(const string text_, string img_path, bool flatten_img = true, BackendType type = MLLM_CPU) {
        string new_text = text_;
        if (!img_path.empty()) {
            auto image_inputs = image_processor.process({std::move(img_path)}, flatten_img);
            auto pixel_values = image_inputs.first;
            auto image_grid_thw = image_inputs.second;
            auto merge_length = image_processor.merge_size * image_processor.merge_size;

            int index = 0; // 跟踪当前使用的网格配置索引
            const int PAD_LEN = IMAGE_PAD.length();
            const int HOLDER_LEN = PLACEHOLDER.length();
            size_t pos = 0;
            // 第一阶段：替换image_pad为placeholder序列
            while (true) {
                // 查找下一个需要替换的位置
                size_t found = new_text.find(IMAGE_PAD, pos);
                if (found == string::npos || index >= image_grid_thw.size()) break;
                // 计算需要插入的placeholder数量
                int product = 1;
                for (int dim : image_grid_thw[index]) {
                    product *= dim;
                }
                int replace_num = product / merge_length;
                // 构建替换字符串
                string replacement;
                replacement.reserve(HOLDER_LEN * replace_num);
                for (int i = 0; i < replace_num; ++i) {
                    replacement += PLACEHOLDER;
                }
                // 执行替换并更新扫描位置
                new_text.replace(found, PAD_LEN, replacement);
                pos = found + replacement.length(); // 跳过已处理部分
                index++;
            }
            // 第二阶段：将placeholder恢复为image_pad
            size_t ph_pos = 0;
            while ((ph_pos = new_text.find(PLACEHOLDER, ph_pos))) {
                if (ph_pos == string::npos) break;
                new_text.replace(ph_pos, HOLDER_LEN, IMAGE_PAD);
                ph_pos += PAD_LEN; // 跳过已替换部分
            }
            auto input_tensor = tokenizer->tokenize(new_text);
            auto image_grid_thw_tensor = vector3d2Tensor({image_grid_thw}, "image_grid_thw");
            return {input_tensor, pixel_values, image_grid_thw_tensor};
        } else{
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
#endif // PROCESSING_Qwen2VL_HPP
