// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <optional>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/utils/StringHelper.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/visual/ImageTransform.hpp"

#include "mllm/models/deepseek_ocr/deepencoder.hpp"
#include "mllm/models/deepseek_ocr/conversation_preprocess.hpp"
#include "mllm/models/deepseek_ocr/tokenization_deepseek_ocr.hpp"
#include "mllm/models/deepseek_ocr/configuration_deepseek_ocr.hpp"

namespace mllm::models::deepseek_ocr {

class DeepseekV2MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU act_;

  int hidden_size_;
  int intermediate_size_;

 public:
  DeepseekV2MLP() = default;

  explicit DeepseekV2MLP(const std::string& name, const DpskOcrConfig& config,
                         const std::optional<int>& hidden_size = std::nullopt,
                         const std::optional<int>& intermediate_size = std::nullopt)
      : nn::Module(name) {
    hidden_size_ = hidden_size.value_or(config.hidden_size);
    intermediate_size_ = intermediate_size.value_or(config.intermediate_size);

    // clang-format off
    gate_proj_ = reg<nn::Linear>("gate_proj", hidden_size_, intermediate_size_, false, config.llm_mlp_linear_impl_type);
    up_proj_ = reg<nn::Linear>("up_proj", hidden_size_, intermediate_size_, false, config.llm_mlp_linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", intermediate_size_, hidden_size_, false, config.llm_mlp_linear_impl_type);
    act_ = reg<nn::SiLU>("act");
    // clang-format on
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {down_proj_(act_(gate_proj_(inputs[0])) * up_proj_(inputs[0]))};
  }
};

class MoEGate final : public nn::Module {
  // FIXME: We may need to support more types
  std::string scoring_func_ = "softmax";
  std::string topk_method_ = "greedy";

  int top_k_;
  int n_routed_experts_;
  float routed_scaling_factor_;
  int n_group_;
  int topk_group_;
  bool norm_topk_prob_;

  nn::Param weight_;

 public:
  MoEGate() = default;

  MoEGate(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    top_k_ = config.num_experts_per_tok;
    n_routed_experts_ = config.n_routed_experts;

    // FIXME: Read from config.json instead of hard-coding
    routed_scaling_factor_ = 1.f;
    norm_topk_prob_ = false;

    n_group_ = config.n_group;
    topk_group_ = config.topk_group;

    weight_ = reg<nn::Param>("weight", getModuleName() + ".weight");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto bsz = hidden_states.size(0);
    auto seq_len = hidden_states.size(1);
    auto h = hidden_states.size(2);

    // Compute gating score
    hidden_states = hidden_states.view({-1, h});
    // hidden_states and weight must in fp32 to keep precision !!!
    auto logits = nn::functional::matmul(hidden_states, weight_.weight(), false, true);
    auto scores = nn::functional::softmax(logits, -1);
    auto [topk_weight, topk_idx] = nn::functional::topk(scores, top_k_, -1, true, false);

    // FIXME: Someone may need to Norm gate to sum 1.
    // FIXME: Someone may need rescale topk_weight by routed_scaling_factor_, but here is hard-code to 1.f

    return {topk_idx, topk_weight};
  }
};

class DeepseekV2MoE final : public nn::Module {
  int num_experts_per_tok_;

  // FIXME: Should not hard-code
  int ep_size_ = 1;
  int experts_per_rank_;
  int n_shared_experts_ = 0;

  nn::ModuleList<DeepseekV2MLP> experts_;
  MoEGate gate_;
  nn::ModuleList<DeepseekV2MoE> shared_experts_;

 public:
  DeepseekV2MoE() = default;

  DeepseekV2MoE(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    num_experts_per_tok_ = config.num_experts_per_tok;
    experts_per_rank_ = config.n_routed_experts;
    n_shared_experts_ = config.n_shared_experts;

    // Init experts
    experts_ = reg<nn::ModuleList<DeepseekV2MLP>>("experts", config.n_routed_experts, config, std::nullopt,
                                                  config.moe_intermediate_size);
    gate_ = reg<MoEGate>("gate", config);

    if (n_shared_experts_ > 0) {
      auto intermediate_size = config.moe_intermediate_size * config.n_shared_experts;
      shared_experts_ =
          reg<nn::ModuleList<DeepseekV2MoE>>("shared_experts", n_shared_experts_, config, std::nullopt, intermediate_size);
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto identity = hidden_states;
    auto orig_shape = hidden_states.shape();
    auto topk_idx = Tensor::nil();
    auto topk_weight = Tensor::nil();
    auto gated_ret = gate_(hidden_states);
    topk_idx = gated_ret[0];
    topk_weight = gated_ret[1];
    hidden_states = hidden_states.view({-1, hidden_states.size(-1)});
    auto flat_topk_idx = topk_idx.view({-1});
    auto y = moeInfer(hidden_states, topk_idx, topk_weight).view(orig_shape);
    if (n_shared_experts_ > 0) { y = y + shared_experts_(identity)[0]; }
    return {y};
  }

 private:
  Tensor moeInfer(const Tensor& x, const Tensor& topk_ids, const Tensor& topk_weights) {
    // TODO
    return Tensor::nil();
  }
};

class DeepseekV2Attention final : public nn::Module {
 public:
  // TODO
};

class DeepseekV2DecoderLayer final : public nn::Module {
 public:
  // TODO
};

class DeepSeekV2Model : public nn::Module {
 protected:
  nn::Embedding embed_tokens_;

 public:
  DeepSeekV2Model() = default;

  explicit DeepSeekV2Model(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    embed_tokens_ = reg<nn::Embedding>("embed_tokens", config.vocab_size, config.hidden_size);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // TODO
    return {};
  }
};

class DeepseekOCRModel final : public DeepSeekV2Model {
  VitModel vision_model_;
  ImageEncoderViT sam_model_;
  MlpProjector projector_;
  nn::Param image_newline_;
  nn::Param view_separator_;
  int n_embed = 1280;

 public:
  DeepseekOCRModel() = default;

  explicit DeepseekOCRModel(const std::string& name, const DpskOcrConfig& config) : DeepSeekV2Model(name, config) {
    sam_model_ = reg<ImageEncoderViT>("sam_model", config);
    vision_model_ = reg<VitModel>("vision_model", config);
    projector_ = reg<MlpProjector>("projector", config);
    image_newline_ = reg<nn::Param>("image_newline", getModuleName() + ".image_newline");
    view_separator_ = reg<nn::Param>("view_seperator", getModuleName() + ".view_seperator");  ///< DeepSeek's typo.
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // FIXME: Just support one image right now.
    // Inputs: should be [input_ids, optional[image_crop], optional[image_ori], optional[images_spatial_crop]]
    auto& input_ids = inputs[0];
    auto patches = inputs.size() > 1 ? inputs[1] : Tensor::nil();
    auto image_ori = inputs.size() > 2 ? inputs[2] : Tensor::nil();
    auto images_spatial_crop = inputs.size() > 3 ? inputs[3] : Tensor::nil();
    auto images_seq_mask = inputs.size() > 4 ? inputs[4] : Tensor::nil();

    // Embedding
    auto inputs_embeds = embed_tokens_(input_ids);

    // We need to process image
    auto images_in_this_batch = Tensor::nil();
    if (patches && image_ori && images_spatial_crop && images_seq_mask) {
      if (nn::functional::sum(patches).item<float>() != 0) {
        // Local features
        auto local_features_1 = sam_model_(patches)[0];
        auto local_features_2 = vision_model_(patches, local_features_1)[0];
        auto local_features = nn::functional::concat(
            {
                local_features_2[{kAll, {1, kAll}}],
                local_features_1.flatten(2).permute({0, 2, 1}),
            },
            -1);
        local_features = projector_(local_features)[0];

        // Global features
        auto global_features_1 = sam_model_(image_ori)[0];
        auto global_features_2 = vision_model_(image_ori, global_features_1)[0];
        auto global_features = nn::functional::concat(
            {
                global_features_2[{kAll, {1, kAll}}],
                global_features_1.flatten(2).permute({0, 2, 1}),
            },
            -1);
        global_features = projector_(global_features)[0];

        print("=====================");
        print("BASE: ", global_features.shape());
        print("PATCHES: ", local_features.shape());
        print("=====================");

        auto hw = global_features.size(1);
        auto n_dim = global_features.size(2);
        auto h = (int)std::sqrt(hw);
        auto w = h;

        auto hw2 = local_features.size(1);
        auto n_dim2 = local_features.size(2);
        auto h2 = (int)std::sqrt(hw2);
        auto w2 = h2;

        MLLM_RT_ASSERT_EQ(images_spatial_crop.dtype(), kInt64);
        int width_crop_num = images_spatial_crop.at<mllm_int64_t>({0, 0});
        int height_crop_num = images_spatial_crop.at<mllm_int64_t>({0, 1});

        global_features = global_features.view({h, w, n_dim});
        global_features = nn::functional::concat(
            {
                global_features,

                // FIXME: This line is in-efficient.
                // pytorch logic: self.image_newline[None, None, :].expand(h, 1, n_dim)
                //
                // Use pytorch like expand instead. Expand will only modified stride, no memory copy involved.
                // But many kernels in mllm's arm backend not use stride as loop step, but calculate itself, so we need to
                // refact it.
                image_newline_.weight().view({1, 1, -1}).repeat(h, 0),
            },
            1);

        global_features = global_features.view({-1, n_dim});

        local_features = local_features.view({height_crop_num, width_crop_num, h2, w2, n_dim2})
                             .permute({0, 2, 1, 3, 4})
                             .view({height_crop_num * h2, width_crop_num * w2, n_dim2});
        local_features = nn::functional::concat(
            {
                local_features,

                // FIXME: This line is in-efficient.
                // pytorch logic: self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)
                //
                // Use pytorch like expand instead. Expand will only modified stride, no memory copy involved.
                // But many kernels in mllm's arm backend not use stride as loop step, but calculate itself, so we need to
                // refact it.
                image_newline_.weight().view({1, 1, -1}).repeat(height_crop_num * h2, 0),
            },
            1);

        local_features = local_features.view({-1, n_dim2});
        auto global_local_features = nn::functional::concat(
            {
                local_features,
                global_features,

                // pytorch logic: self.view_seperator[None, :]
                view_separator_.weight().view({1, -1}),
            },
            0);
        images_in_this_batch = global_local_features;
      } else {
        auto global_features_1 = sam_model_(image_ori)[0];
        auto global_features_2 = vision_model_(image_ori, global_features_1)[0];
        auto global_features = nn::functional::concat(
            {
                global_features_2[{kAll, {1, kAll}}],
                global_features_1.flatten(2).permute({0, 2, 1}),
            },
            -1);

        global_features = projector_(global_features)[0];

        print("=====================");
        print("BASE: ", global_features.shape());
        print("NO PATCHES");
        print("=====================");

        auto hw = global_features.size(1);
        auto n_dim = global_features.size(2);
        auto h = (int)std::sqrt(hw);
        auto w = h;

        global_features = global_features.view({h, w, n_dim});
        global_features = nn::functional::concat(
            {
                global_features,

                // FIXME: This line is in-efficient.
                // pytorch logic: self.image_newline[None, None, :].expand(h, 1, n_dim)
                //
                // Use pytorch like expand instead. Expand will only modified stride, no memory copy involved.
                // But many kernels in mllm's arm backend not use stride as loop step, but calculate itself, so we need to
                // refact it.
                image_newline_.weight().view({1, 1, -1}).repeat(h, 0),
            },
            1);

        global_features = global_features.view({-1, n_dim});

        auto global_local_features = nn::functional::concat(
            {
                global_features,
                view_separator_.weight().view({1, -1}),
            },
            0);

        images_in_this_batch = global_local_features;
      }
    }

    // Scatter copy.
    if (images_in_this_batch) { nn::functional::maskedScatter(inputs_embeds, images_seq_mask, images_in_this_batch); }

    // Normal forward with text and embedded image
    // TODO

    return {};
  }
};

class DeepseekOCRForCausalLM final : public nn::Module, public ARGeneration {
  DeepseekOCRModel model_;
  nn::Linear lm_head_;

 public:
  DeepseekOCRForCausalLM() = default;

  explicit DeepseekOCRForCausalLM(const DpskOcrConfig& config) {
    model_ = reg<DeepseekOCRModel>("model", config);
    lm_head_ = reg<nn::Linear>("lm_head", config.hidden_size, config.vocab_size, false, config.lm_head_linear_impl_type);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; }

  void infer(DpskOcrTokenizer& tokenizer, const std::string& prompt, const std::string& image_fp,
             const std::string& output_path, int base_size = 1024, int image_size = 640, bool crop_mode = true) {
    // Initialize template
    initializeTemplates();

    namespace fs = std::filesystem;
    fs::path out_path(output_path);
    fs::create_directories(out_path);
    fs::create_directories(out_path / "images");

    nlohmann::json conversations;
    if (!prompt.empty() && !image_fp.empty()) {
      conversations = nlohmann::json::array();
      conversations.push_back({{"role", "<|User|>"}, {"content", prompt}, {"images", nlohmann::json::array({image_fp})}});
      conversations.push_back({{"role", "<|Assistant|>"}, {"content", ""}});
    } else if (!prompt.empty()) {
      conversations = nlohmann::json::array();
      conversations.push_back({{"role", "<|User|>"}, {"content", prompt}});
      conversations.push_back({{"role", "<|Assistant|>"}, {"content", ""}});
    } else {
      // Prompt should not be empty
      MLLM_RT_ASSERT_EQ(prompt.empty(), false);
    }

    auto processed_prompt = formatMessages(conversations, "plain", "");

    // Global constant define
    const int PATCH_SIZE = 16;
    const int DOWN_SAMPLE_RATIO = 4;
    const std::string IMAGE_TOKEN = "<image>";
    const int64_t IMAGE_TOKEN_ID = 128815;

    // Global states
    int valid_img_tokens = 0;
    float ratio = 1.f;

    // Load image
    auto images = loadImages(conversations);

    auto w = images[0].w();
    auto h = images[0].h();
    ratio = 1 - (float)((std::max(w, h) - std::min(w, h)) / (float)(std::max(w, h)));

    // Image transform infra
    auto image_transform = BasicImageTransform(std::nullopt, std::nullopt, /*mean=*/std::vector<float>{0.5, 0.5, 0.5},
                                               /*std=*/std::vector<float>{0.5, 0.5, 0.5});

    // Split text with IMAGE_TOKEN
    // Like what python does: text_splits = prompt.split(image_token)
    auto text_splits = mllm::splitString(processed_prompt, IMAGE_TOKEN);

    // Processed states
    std::vector<int64_t> tokenized_str;
    std::vector<float> images_seq_mask;
    std::vector<Tensor> images_list;
    std::vector<Tensor> images_crop_list;
    std::vector<std::tuple<int, int>> images_spatial_crop;

    // text_splits's length should be greater than images' length.
    // text_splits.size() - images.size() = 1
    for (int idx = 0; idx < std::min(images.size(), text_splits.size()); ++idx) {
      auto tokenized_sep = tokenizer.encode(text_splits[idx]);
      tokenized_str.insert(tokenized_str.end(), tokenized_sep.begin(), tokenized_sep.end());
      for (int _i = 0; _i < tokenized_sep.size(); ++_i) {
        images_seq_mask.emplace_back(0);  // emplace_back(false)
      }

      // Get image in this loop
      auto image = images[idx];
      std::tuple<int, int> crop_ratio;
      std::vector<Image> images_crop_raw;

      // Processing Image
      if (crop_mode) {
        if (image.h() <= 640 && image.w() <= 640) {
          crop_ratio = {1, 1};
        } else {
          if (crop_mode) {
            auto p = dynamicPreprocess(image);
            images_crop_raw = p.first;
            crop_ratio = p.second;
          } else {
            crop_ratio = {1, 1};
          }
        }

        // color=tuple(int(x * 255) for x in image_transform.mean
        auto global_view = image.pad(base_size, base_size, (int)(255 * 0.5), (int)(255 * 0.5), (int)(255 * 0.5));

        if (base_size == 1024) {
          valid_img_tokens += (int)(256 * ratio);
        } else if (base_size == 1280) {
          valid_img_tokens += (int)(400 * ratio);
        } else {
          MLLM_RT_ASSERT(false);
        }

        images_list.emplace_back(image_transform(global_view));

        auto [width_crop_num, height_crop_num] = crop_ratio;
        images_spatial_crop.emplace_back(width_crop_num, height_crop_num);

        // Processing crops
        if (width_crop_num > 1 || height_crop_num > 1) {
          for (const auto& _i : images_crop_raw) { images_crop_list.emplace_back(image_transform(_i)); }
        }

        // Check if image_size is 640
        valid_img_tokens += images_crop_list.size() * 100;

        // Compute query
        auto num_queries = std::ceil((image_size / PATCH_SIZE) / DOWN_SAMPLE_RATIO);
        auto num_queries_base = std::ceil((base_size / PATCH_SIZE) / DOWN_SAMPLE_RATIO);

        // Do python logic below:
        // tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
        // tokenized_image += [image_token_id]
        std::vector<int64_t> tokenized_image;
        tokenized_image.reserve((num_queries_base + 1) * num_queries_base + 1);
        for (int i = 0; i < num_queries_base; ++i) {
          tokenized_image.insert(tokenized_image.end(), num_queries_base, IMAGE_TOKEN_ID);
          tokenized_image.push_back(IMAGE_TOKEN_ID);
        }
        tokenized_image.push_back(IMAGE_TOKEN_ID);

        if (width_crop_num > 1 || height_crop_num > 1) {
          for (int h = 0; h < num_queries * height_crop_num; ++h) {
            tokenized_image.insert(tokenized_image.end(), num_queries * width_crop_num, IMAGE_TOKEN_ID);
            tokenized_image.push_back(IMAGE_TOKEN_ID);
          }
        }

        tokenized_str.insert(tokenized_str.end(), tokenized_image.begin(), tokenized_image.end());
        for (int _i = 0; _i < tokenized_image.size(); ++_i) { images_seq_mask.emplace_back(true); }
      } else {
        NYI("crop_mode = false is not supported yet.");
      }
    }

    // Processing last text split
    auto tokenized_sep = tokenizer.encode(text_splits.back());
    tokenized_str.insert(tokenized_str.end(), tokenized_sep.begin(), tokenized_sep.end());
    images_seq_mask.insert(images_seq_mask.end(), tokenized_sep.size(), false);

    // Add bos token
    // bos_id = 0
    // tokenized_str = [bos_id] + tokenized_str
    // images_seq_mask = [False] + images_seq_mask
    tokenized_str.insert(tokenized_str.begin(), 0);
    images_seq_mask.insert(images_seq_mask.begin(), false);

    // Prepare Tensor to DeepSeek-OCR Model
    auto input_ids = Tensor::fromVector(tokenized_str, {1, (int32_t)tokenized_str.size()}, kInt64);
    auto images_seq_mask_tensor = Tensor::fromVector(images_seq_mask, {1, (int32_t)images_seq_mask.size()}, kFloat32);
    auto images_ori_tensor = Tensor::nil();
    auto images_spatial_crop_tensor = Tensor::nil();
    auto images_crop_tensor = Tensor::nil();
    if (images_list.empty()) {
      images_ori_tensor = Tensor::zeros({1, 3, image_size, image_size});
      images_spatial_crop_tensor = Tensor::zeros({1, 2}, kInt64);
      images_crop_tensor = Tensor::zeros({1, 3, base_size, base_size});
    } else {
      images_ori_tensor = nn::functional::stack(images_list, 0);
      images_spatial_crop_tensor = Tensor::zeros({(int32_t)images_spatial_crop.size(), 2}, kInt64);
      auto _ptr = images_spatial_crop_tensor.ptr<mllm_int64_t>();
      for (int _i = 0; _i < images_spatial_crop.size(); ++_i) {
        auto [l, h] = images_spatial_crop[_i];
        _ptr[2 * _i + 0] = l;
        _ptr[2 * _i + 1] = h;
      }
      if (!images_crop_list.empty()) {
        images_crop_tensor = nn::functional::stack(images_crop_list, 0);
      } else {
        images_crop_tensor = Tensor::zeros({1, 3, base_size, base_size});
      }
    }

    // Run model. Use generate
    // TODO

    // Post process data
    // TODO
  }
};

}  // namespace mllm::models::deepseek_ocr
