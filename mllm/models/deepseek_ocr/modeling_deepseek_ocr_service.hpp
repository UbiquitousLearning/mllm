#pragma once

#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>

#include "mllm/engine/service/Session.hpp"
#include "mllm/engine/prefix_cache/Cache.hpp" 

#include "mllm/models/deepseek_ocr/configuration_deepseek_ocr.hpp"
#include "mllm/models/deepseek_ocr/tokenization_deepseek_ocr.hpp"
#include "mllm/models/deepseek_ocr/conversation_preprocess.hpp"
#include "mllm/models/deepseek_ocr/modeling_deepseek_ocr.hpp"
#include "mllm/preprocessor/visual/ImageTransform.hpp"

namespace mllm::models::deepseek_ocr {

class DeepseekOCRSession final : public ::mllm::service::Session {
public:
    DeepseekOCRSession() {
        initializeTemplates();
    }

    void fromPreTrain(const std::string& model_path) override {
        namespace fs = std::filesystem;
        fs::path root = fs::path(model_path).lexically_normal();
        fs::path config_file = root / "config.json";
        fs::path model_file = root / "model.mllm";
        fs::path tokenizer_file = root / "tokenizer.json";

        if (!fs::exists(config_file)) throw std::runtime_error(config_file.string() + " not found");
        if (!fs::exists(model_file)) throw std::runtime_error(model_file.string() + " not found");
        if (!fs::exists(tokenizer_file)) throw std::runtime_error(tokenizer_file.string() + " not found");

        printf("[C++ Service] Loading DeepSeek-OCR model from: %s\n", model_path.c_str());

        config_ = DpskOcrConfig(config_file.string());
        model_ = std::make_shared<DeepseekOCRForCausalLM>(config_);
        model_->load(load(model_file.string(), ModelFileVersion::kV2));
        tokenizer_ = std::make_shared<DpskOcrTokenizer>(tokenizer_file.string());

        printf("[C++ Service] DeepSeek-OCR model loaded successfully.\n");
    }


    void streamGenerate(const nlohmann::json& request,
                          const std::function<void(const nlohmann::json&, bool)>& callback) override {
        
        printf("[C++ Service] Clearing KV Cache for new OCR request (via clearCache)...\n");
        model_->kvCache().clearCache(); 


        const int base_size = 512;
        const int image_size = 512;

        const bool crop_mode = true;
        const int PATCH_SIZE = 16;
        const int DOWN_SAMPLE_RATIO = 4;
        const std::string IMAGE_TOKEN = "<image>";
        const int64_t IMAGE_TOKEN_ID = 128815; 

        auto image_transform = BasicImageTransform(std::nullopt, std::nullopt, 
                                                     std::vector<float>{0.5, 0.5, 0.5},
                                                     std::vector<float>{0.5, 0.5, 0.5});
        
        auto images = loadImages(request["messages"]);
        if (images.empty()) {
            throw std::runtime_error("DeepSeek-OCR requires an image, but none were found in request.");
        }

        std::string user_text;
        const auto& messages = request["messages"];
        for (int i = messages.size() - 1; i >= 0; --i) {
            const auto& msg = messages[i];
            if (msg.value("role", "") == "user" || msg.value("role", "") == "<|User|>") {
                if (msg.contains("content") && msg["content"].is_string()) {
                    user_text = msg["content"].get<std::string>();
                    break; 
                }
            }
        }

        std::vector<int64_t> tokenized_str;
        std::vector<int8_t> images_seq_mask;
        std::vector<Tensor> images_list;
        std::vector<Tensor> images_crop_list;
        std::vector<std::tuple<int, int>> images_spatial_crop;
        
        auto tokenized_sep_before = tokenizer_->encode("");
        tokenized_str.insert(tokenized_str.end(), tokenized_sep_before.begin(), tokenized_sep_before.end());
        images_seq_mask.insert(images_seq_mask.end(), tokenized_sep_before.size(), (int8_t)false);

        auto image = images[0];
        std::tuple<int, int> crop_ratio;
        std::vector<Image> images_crop_raw; 

        if (image.h() <= 640 && image.w() <= 640) {
            printf("[C++ Service] Image is small (%dx%d). Skipping dynamic preprocess.\n", image.w(), image.h());
            crop_ratio = {1, 1};
        } else {
            printf("[C++ Service] Processing large image with dynamicPreprocess (target size: %d)...\n", image_size);
            auto p = dynamicPreprocess(image, 2, 9, image_size, false); 
            images_crop_raw = p.first;
            crop_ratio = p.second;
            printf("[C++ Service] Image processed. Grid: %d x %d (%d tiles)\n", 
                   std::get<0>(crop_ratio), std::get<1>(crop_ratio), (int)images_crop_raw.size());
        }

        auto global_view = image.pad(base_size, base_size, (int)(255 * 0.5), (int)(255 * 0.5), (int)(255 * 0.5));
        images_list.emplace_back(image_transform(global_view)); 

        auto [width_crop_num, height_crop_num] = crop_ratio;
        images_spatial_crop.emplace_back(width_crop_num, height_crop_num);

        if (width_crop_num > 1 || height_crop_num > 1) {
            for (const auto& _i : images_crop_raw) {
                images_crop_list.emplace_back(image_transform(_i));
            }
        }
    
        auto num_queries_base = std::ceil((base_size / PATCH_SIZE) / DOWN_SAMPLE_RATIO);
        std::vector<int64_t> tokenized_image;
        tokenized_image.reserve((num_queries_base + 1) * num_queries_base + 1);
        for (int i = 0; i < num_queries_base; ++i) {
            tokenized_image.insert(tokenized_image.end(), num_queries_base, IMAGE_TOKEN_ID);
            tokenized_image.push_back(IMAGE_TOKEN_ID);
        }
        tokenized_image.push_back(IMAGE_TOKEN_ID);

        if (width_crop_num > 1 || height_crop_num > 1) { 
            auto num_queries = std::ceil((image_size / PATCH_SIZE) / DOWN_SAMPLE_RATIO);
            for (int h = 0; h < num_queries * height_crop_num; ++h) {
                tokenized_image.insert(tokenized_image.end(), num_queries * width_crop_num, IMAGE_TOKEN_ID);
                tokenized_image.push_back(IMAGE_TOKEN_ID);
            }
        }

        tokenized_str.insert(tokenized_str.end(), tokenized_image.begin(), tokenized_image.end());
        images_seq_mask.insert(images_seq_mask.end(), tokenized_image.size(), (int8_t)true);
        
        std::string final_prompt_text = "\n<|grounding|>" + user_text;
        auto tokenized_sep_after = tokenizer_->encode(final_prompt_text);

        tokenized_str.insert(tokenized_str.end(), tokenized_sep_after.begin(), tokenized_sep_after.end());
        images_seq_mask.insert(images_seq_mask.end(), tokenized_sep_after.size(), (int8_t)false);

        tokenized_str.insert(tokenized_str.begin(), config_.bos_token_id);
        images_seq_mask.insert(images_seq_mask.begin(), (int8_t)false);

        auto input_ids = Tensor::fromVector(tokenized_str, {1, (int32_t)tokenized_str.size()}, kInt64);
        auto images_seq_mask_tensor = Tensor::fromVector(images_seq_mask, {1, (int32_t)images_seq_mask.size()}, kInt8);
        auto images_ori_tensor = nn::functional::stack(images_list, 0); 
        
        auto images_spatial_crop_tensor = Tensor::zeros({(int32_t)images_spatial_crop.size(), 2}, kInt64);
        auto* _ptr = images_spatial_crop_tensor.ptr<mllm_int64_t>();
        for (int _i = 0; _i < images_spatial_crop.size(); ++_i) {
            auto [l, h] = images_spatial_crop[_i];
            _ptr[2 * _i + 0] = l;
            _ptr[2 * _i + 1] = h;
        }

        Tensor images_crop_tensor;
        if (!images_crop_list.empty()) {
            images_crop_tensor = nn::functional::stack(images_crop_list, 0); 
        } else {
            images_crop_tensor = Tensor::zeros({1, 3, image_size, image_size});
        }
        
        ARGenerationOutputPast input;
        input["sequence"] = input_ids;
        input["patches"] = images_crop_tensor;
        input["image_ori"] = images_ori_tensor;
        input["images_spatial_crop"] = images_spatial_crop_tensor;
        input["images_seq_mask"] = images_seq_mask_tensor;

        ARGenerationArgs args;
        args["kv_cache"] = mllm::AnyValue(&model_->kvCache());
        
        args["temperature"] = request.value("temperature", 1.0f);
        args["top_k"] = request.value("top_k", 0);
        args["top_p"] = request.value("top_p", 0.0f);
        args["max_length"] = request.value("max_length", 1024);
        args["do_sample"] = request.value("do_sample", false);
        
        model_->streamGenerate(input, args, [this, &callback](int64_t idx) {
            bool finished = false;
            std::string ret_token;

            if (idx == model_->eosTokenId()) {
                finished = true;
                ret_token = "";
            } else {
                finished = false;
                ret_token = tokenizer_->decode({idx});
            }

            callback(ret_token, finished);
        });
    }

private:
    std::shared_ptr<DeepseekOCRForCausalLM> model_;
    std::shared_ptr<DpskOcrTokenizer> tokenizer_;
    DpskOcrConfig config_;
};

}