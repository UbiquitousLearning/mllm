// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/models/qwen2vl/configuration_qwen2vl.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl_traceable.hpp>

namespace qwen2vl_qnn_aot {

class PatchEmbedLinear final : public mllm::nn::Module {
  int32_t patch_dim_ = 0;
  int32_t embed_dim_ = 0;

  mllm::nn::Linear proj_;

 public:
  PatchEmbedLinear() = default;

  PatchEmbedLinear(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg) : mllm::nn::Module(name) {
    patch_dim_ = cfg.visual_in_chans * cfg.visual_temporal_patch_size * cfg.visual_patch_size * cfg.visual_patch_size;
    embed_dim_ = cfg.visual_embed_dim;
    proj_ = reg<mllm::nn::Linear>("proj", patch_dim_, embed_dim_, false, mllm::aops::LinearImplTypes::kDefault);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    hidden_states = hidden_states.view({-1, patch_dim_}, true);
    hidden_states = proj_(hidden_states).view({-1, embed_dim_}, true);
    return {hidden_states};
  }
};

class VisionMlpPrimitiveQuickGELU final : public mllm::nn::Module {
  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;

  mllm::nn::Linear fc_1_;
  mllm::nn::Linear fc_2_;

 public:
  VisionMlpPrimitiveQuickGELU() = default;

  VisionMlpPrimitiveQuickGELU(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    dim_ = cfg.visual_embed_dim;
    hidden_dim_ = cfg.visual_embed_dim * cfg.visual_mlp_ratio;
    fc_1_ = reg<mllm::nn::Linear>("fc1", dim_, hidden_dim_, true, cfg.linear_impl_type);
    fc_2_ = reg<mllm::nn::Linear>("fc2", hidden_dim_, dim_, true, cfg.linear_impl_type);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto x = fc_1_(inputs[0]);
    x = x * mllm::nn::functional::sigmoid(x * 1.702f);
    return {fc_2_(x)};
  }
};

class VisionAttentionMaskedAOTRewrite final : public mllm::nn::Module {
  int32_t dim_ = 0;
  int32_t num_heads_ = 0;
  int32_t head_dim_ = 0;

  mllm::nn::Linear qkv_;
  mllm::nn::Linear proj_;
  mllm::nn::Softmax softmax_;

 public:
  VisionAttentionMaskedAOTRewrite() = default;

  VisionAttentionMaskedAOTRewrite(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    dim_ = cfg.visual_embed_dim;
    num_heads_ = cfg.visual_num_heads;
    head_dim_ = dim_ / num_heads_;

    qkv_ = reg<mllm::nn::Linear>("qkv", dim_, dim_ * 3, true, cfg.linear_impl_type);
    proj_ = reg<mllm::nn::Linear>("proj", dim_, dim_, true, cfg.linear_impl_type);
    softmax_ = reg<mllm::nn::Softmax>("softmax", -1);
  }

  mllm::Tensor applyVisionRoPEPrimitive(mllm::Tensor x, mllm::Tensor visual_embedding_sin, mllm::Tensor visual_embedding_cos) {
    const int32_t half_dim = head_dim_ / 2;
    auto x1 = x.slice({mllm::kAll, mllm::kAll, mllm::kAll, {mllm::kAll, half_dim}}, true);
    auto x2 = x.slice({mllm::kAll, mllm::kAll, mllm::kAll, {half_dim, mllm::kAll}}, true);
    auto sin = visual_embedding_sin;
    auto cos = visual_embedding_cos;
    if (sin.rank() == 2) {
      sin = sin.view({1, -1, 1, half_dim}, true);
      cos = cos.view({1, -1, 1, half_dim}, true);
    } else {
      MLLM_RT_ASSERT_EQ(sin.rank(), 4);
      MLLM_RT_ASSERT_EQ(cos.rank(), 4);
    }

    auto y1 = x1 * cos + (-(x2 * sin));
    auto y2 = x1 * sin + x2 * cos;
    return mllm::nn::functional::concat({y1, y2}, -1);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];
    auto attention_mask = inputs.size() > 3 ? inputs[3] : mllm::Tensor::nil();

    auto qkv_states = qkv_(hidden_states).view({-1, 3, num_heads_, head_dim_}, true);
    auto query_states = qkv_states.slice({mllm::kAll, {0, 1}, mllm::kAll, mllm::kAll}, true).transpose(0, 1);
    auto key_states = qkv_states.slice({mllm::kAll, {1, 2}, mllm::kAll, mllm::kAll}, true).transpose(0, 1);
    auto value_states = qkv_states.slice({mllm::kAll, {2, 3}, mllm::kAll, mllm::kAll}, true).transpose(0, 1);

    query_states = applyVisionRoPEPrimitive(query_states, visual_embedding_sin, visual_embedding_cos);
    key_states = applyVisionRoPEPrimitive(key_states, visual_embedding_sin, visual_embedding_cos);

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    auto attn = mllm::nn::functional::matmul(query_states, key_states, false, true) * (1.f / std::sqrt(static_cast<float>(head_dim_)));
    if (!attention_mask.isNil()) { attn = attn + attention_mask; }
    attn = softmax_(attn);

    auto attn_output = mllm::nn::functional::matmul(attn, value_states);
    attn_output = attn_output.transpose(1, 2).view({-1, dim_}, true);
    return {proj_(attn_output)};
  }
};

class Qwen2VLVisionBlockAOTRewrite final : public mllm::nn::Module {
  mllm::nn::LayerNorm norm1_;
  mllm::nn::LayerNorm norm2_;

  VisionAttentionMaskedAOTRewrite attn_;
  VisionMlpPrimitiveQuickGELU mlp_;

 public:
  Qwen2VLVisionBlockAOTRewrite() = default;

  Qwen2VLVisionBlockAOTRewrite(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    norm1_ = reg<mllm::nn::LayerNorm>("norm1", std::vector<int32_t>{cfg.visual_embed_dim}, true, true, 1e-6);
    norm2_ = reg<mllm::nn::LayerNorm>("norm2", std::vector<int32_t>{cfg.visual_embed_dim}, true, true, 1e-6);
    attn_ = reg<VisionAttentionMaskedAOTRewrite>("attn", cfg);
    mlp_ = reg<VisionMlpPrimitiveQuickGELU>("mlp", cfg);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];
    auto attention_mask = inputs.size() > 3 ? inputs[3] : mllm::Tensor::nil();

    if (attention_mask.isNil()) {
      hidden_states = hidden_states + attn_(norm1_(hidden_states), visual_embedding_sin, visual_embedding_cos)[0];
    } else {
      hidden_states =
          hidden_states + attn_(norm1_(hidden_states), visual_embedding_sin, visual_embedding_cos, attention_mask)[0];
    }
    hidden_states = hidden_states + mlp_(norm2_(hidden_states))[0];
    return {hidden_states};
  }
};

class Qwen2VisionTransformerPretrainedModelAOTRewrite final : public mllm::nn::Module {
  PatchEmbedLinear patch_embed_;
  mllm::models::qwen2vl::PatchMerger patch_merger_;
  mllm::nn::ModuleList<Qwen2VLVisionBlockAOTRewrite> blocks_;
  int32_t start_block_ = 0;
  int32_t active_blocks_ = -1;
  bool skip_merger_ = false;
  bool skip_patch_embed_ = false;

 public:
  Qwen2VisionTransformerPretrainedModelAOTRewrite() = default;

  Qwen2VisionTransformerPretrainedModelAOTRewrite(const std::string& name,
                                                  const mllm::models::qwen2vl::Qwen2VLConfig& cfg,
                                                  int32_t start_block = 0,
                                                  int32_t active_blocks = -1,
                                                  bool skip_merger = false,
                                                  bool skip_patch_embed = false)
      : mllm::nn::Module(name) {
    start_block_ = start_block;
    active_blocks_ = active_blocks;
    skip_merger_ = skip_merger;
    skip_patch_embed_ = skip_patch_embed;
    patch_embed_ = reg<PatchEmbedLinear>("patch_embed", cfg);
    patch_merger_ = reg<mllm::models::qwen2vl::PatchMerger>("merger", cfg);
    blocks_ = reg<mllm::nn::ModuleList<Qwen2VLVisionBlockAOTRewrite>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto embedding_sin = inputs[1];
    auto embedding_cos = inputs[2];
    auto attention_mask = inputs.size() > 3 ? inputs[3] : mllm::Tensor::nil();

    if (!skip_patch_embed_) { hidden_states = patch_embed_(hidden_states)[0]; }
    auto num_blocks = active_blocks_ < 0 ? static_cast<int32_t>(blocks_.list().size()) : active_blocks_;
    MLLM_RT_ASSERT(start_block_ >= 0);
    MLLM_RT_ASSERT(num_blocks >= 0);
    MLLM_RT_ASSERT(start_block_ + num_blocks <= static_cast<int32_t>(blocks_.list().size()));
    for (int32_t i = 0; i < num_blocks; ++i) {
      if (attention_mask.isNil()) {
        hidden_states = blocks_.list()[start_block_ + i](hidden_states, embedding_sin, embedding_cos)[0];
      } else {
        hidden_states = blocks_.list()[start_block_ + i](hidden_states, embedding_sin, embedding_cos, attention_mask)[0];
      }
    }
    if (!skip_merger_) { hidden_states = patch_merger_(hidden_states)[0]; }

    return {hidden_states};
  }
};

struct VisualSegment {
  std::string graph_name;
  int32_t start_block = 0;
  int32_t visual_blocks = 0;
  bool skip_patch_embed = false;
  bool skip_merger = false;
  std::vector<int32_t> input_shape;
  std::string ir_name;
};

struct VisualBucketGrid {
  int32_t grid_h = 0;
  int32_t grid_w = 0;

  [[nodiscard]] int32_t patchTokens() const { return grid_h * grid_w; }
};

inline std::vector<VisualBucketGrid> parseVisualBucketGrids(const std::string& text) {
  std::vector<VisualBucketGrid> buckets;
  if (text.empty()) { return buckets; }

  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }), item.end());
    if (item.empty()) { continue; }

    auto sep = item.find('x');
    if (sep == std::string::npos) { sep = item.find('X'); }
    if (sep == std::string::npos) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid visual bucket grid '{}', expected HxW.", item);
    }

    const int32_t grid_h = std::stoi(item.substr(0, sep));
    const int32_t grid_w = std::stoi(item.substr(sep + 1));
    if (grid_h <= 0 || grid_w <= 0 || grid_h % 2 != 0 || grid_w % 2 != 0) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                      "Invalid visual bucket grid '{}': grid_h/grid_w must be positive even numbers.",
                      item);
    }
    buckets.push_back({grid_h, grid_w});
  }
  return buckets;
}

inline std::vector<int32_t> uniqueVisualBucketPatchTokens(const std::vector<VisualBucketGrid>& buckets) {
  std::vector<int32_t> tokens;
  std::unordered_set<int32_t> seen;
  for (const auto& bucket : buckets) {
    const int32_t patch_tokens = bucket.patchTokens();
    if (seen.insert(patch_tokens).second) { tokens.push_back(patch_tokens); }
  }
  std::sort(tokens.begin(), tokens.end());
  return tokens;
}

inline std::string visualGraphSuffixForPatchTokens(int32_t patch_tokens) { return "_s" + std::to_string(patch_tokens); }

inline void reshapePatchEmbedConv3DWeightForLinear(const mllm::ParameterFile::ptr_t& params,
                                                   const mllm::models::qwen2vl::Qwen2VLConfig& cfg) {
  const std::string weight_name = "visual.patch_embed.proj.weight";
  if (!params->has(weight_name)) { MLLM_ERROR_EXIT(mllm::ExitCode::kIOError, "Missing {}", weight_name); }

  const int32_t patch_dim = cfg.visual_in_chans * cfg.visual_temporal_patch_size * cfg.visual_patch_size * cfg.visual_patch_size;
  auto weight = params->pull(weight_name);
  MLLM_RT_ASSERT_EQ(weight.numel(), cfg.visual_embed_dim * patch_dim);

  params->remove(weight_name);
  params->push(weight_name, weight.view({cfg.visual_embed_dim, patch_dim}));
}

inline std::vector<VisualSegment> makeVisualBundleSegments(const std::string& layout,
                                                           int32_t visual_depth,
                                                           int32_t visual_patch_tokens,
                                                           int32_t patch_flat_dim,
                                                           const mllm::models::qwen2vl::Qwen2VLConfig& cfg,
                                                           const std::string& graph_suffix = "") {
  std::vector<VisualSegment> segments;
  if (layout == "single") {
    segments = {
        {"visual_full" + graph_suffix, 0, visual_depth, false, false, {visual_patch_tokens, patch_flat_dim}, "full" + graph_suffix},
    };
  } else if (layout == "6x8") {
    segments = {
        {"visual_patch_embed" + graph_suffix, 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed" + graph_suffix},
        {"visual_blocks_0_8" + graph_suffix, 0, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_0_8" + graph_suffix},
        {"visual_blocks_8_16" + graph_suffix, 8, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_8_16" + graph_suffix},
        {"visual_blocks_16_24" + graph_suffix, 16, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_16_24" + graph_suffix},
        {"visual_blocks_24_32" + graph_suffix, 24, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_24_32" + graph_suffix},
        {"visual_merger" + graph_suffix, 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger" + graph_suffix},
    };
  } else if (layout == "tail4") {
    segments = {
        {"visual_patch_embed" + graph_suffix, 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed" + graph_suffix},
        {"visual_blocks_0_8" + graph_suffix, 0, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_0_8" + graph_suffix},
        {"visual_blocks_8_16" + graph_suffix, 8, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_8_16" + graph_suffix},
        {"visual_blocks_16_20" + graph_suffix, 16, 4, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_16_20" + graph_suffix},
        {"visual_blocks_20_24" + graph_suffix, 20, 4, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_20_24" + graph_suffix},
        {"visual_blocks_24_28" + graph_suffix, 24, 4, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_24_28" + graph_suffix},
        {"visual_blocks_28_32" + graph_suffix, 28, 4, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_28_32" + graph_suffix},
        {"visual_merger" + graph_suffix, 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger" + graph_suffix},
    };
  } else if (layout == "early2") {
    segments = {
        {"visual_patch_embed" + graph_suffix, 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed" + graph_suffix},
        {"visual_blocks_0_2" + graph_suffix, 0, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_0_2" + graph_suffix},
        {"visual_blocks_2_4" + graph_suffix, 2, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_2_4" + graph_suffix},
        {"visual_blocks_4_6" + graph_suffix, 4, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_4_6" + graph_suffix},
        {"visual_blocks_6_8" + graph_suffix, 6, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_6_8" + graph_suffix},
        {"visual_blocks_8_16" + graph_suffix, 8, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_8_16" + graph_suffix},
        {"visual_blocks_16_24" + graph_suffix, 16, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_16_24" + graph_suffix},
        {"visual_blocks_24_32" + graph_suffix, 24, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_24_32" + graph_suffix},
        {"visual_merger" + graph_suffix, 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger" + graph_suffix},
    };
  } else if (layout == "block1") {
    segments.push_back({"visual_patch_embed" + graph_suffix, 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed" + graph_suffix});
    for (int32_t i = 0; i < visual_depth; ++i) {
      const auto graph_name = "visual_blocks_" + std::to_string(i) + "_" + std::to_string(i + 1) + graph_suffix;
      const auto ir_name = "blocks_" + std::to_string(i) + "_" + std::to_string(i + 1) + graph_suffix;
      segments.push_back({graph_name, i, 1, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, ir_name});
    }
    segments.push_back({"visual_merger" + graph_suffix, 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger" + graph_suffix});
  } else {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Unsupported visual bundle layout: {}", layout);
  }
  return segments;
}

}  // namespace qwen2vl_qnn_aot
