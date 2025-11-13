// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"

namespace mllm::models::minicpmo {

// SigLIP Vision Embeddings
class SiglipVisionEmbeddings final : public nn::Module {
  int32_t embed_dim_;
  int32_t image_size_;
  int32_t patch_size_;
  int32_t num_patches_per_side_;
  int32_t num_patches_;

  nn::Conv2D patch_embedding_;
  nn::Embedding position_embedding_;

 public:
  SiglipVisionEmbeddings() = default;

  inline SiglipVisionEmbeddings(const std::string& name, const MiniCPMOConfig& config) : nn::Module(name) {
    embed_dim_ = config.vision_hidden_size;
    image_size_ = config.vision_image_size;
    patch_size_ = config.vision_patch_size;
    num_patches_per_side_ = image_size_ / patch_size_;
    num_patches_ = num_patches_per_side_ * num_patches_per_side_;

    patch_embedding_ =
        reg<nn::Conv2D>("patch_embedding", 3, embed_dim_, Tensor::shape_t{patch_size_, patch_size_},
                        Tensor::shape_t{patch_size_, patch_size_}, Tensor::shape_t{0, 0}, Tensor::shape_t{1, 1}, true);
    position_embedding_ = reg<nn::Embedding>("position_embedding", num_patches_, embed_dim_);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto pixel_values = inputs[0];
    auto tgt_sizes = inputs.size() > 1 ? inputs[1] : Tensor::nil();
    auto patch_attention_mask = inputs.size() > 2 ? inputs[2] : Tensor::nil();

    auto batch_size = pixel_values.shape()[0];

    auto patch_embeds = patch_embedding_(pixel_values);  // Patch embedding: [B, C, H, W] -> [B, embed_dim, 1, H*W]

    auto embeddings = patch_embeds.squeeze(2).transpose(1, 2);  // [B, embed_dim, 1, H*W] -> [B, H*W, embed_dim]

    // Create position embeddings
    if (!tgt_sizes.isNil() && !patch_attention_mask.isNil()) {
      auto max_im_h = pixel_values.shape()[2];
      auto max_im_w = pixel_values.shape()[3];
      auto max_nb_patches_h = max_im_h / patch_size_;
      auto max_nb_patches_w = max_im_w / patch_size_;

      // Create boundaries like torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
      std::vector<float> boundaries;
      float step = 1.0f / static_cast<float>(num_patches_per_side_);
      for (int i = 1; i < num_patches_per_side_; ++i) { boundaries.push_back(i * step); }

      // Create position_ids tensor - using the max_patches from patch_attention_mask shape
      auto max_patches = patch_attention_mask.shape()[2];
      auto position_ids = Tensor::empty({batch_size, max_patches}, kInt64).alloc();
      // Initialize to zeros
      for (int b = 0; b < batch_size; b++) {
        for (int p = 0; p < max_patches; p++) { position_ids.at<int64_t>({b, p}) = 0; }
      }

      // Fill position ids based on patch grid and attention mask
      for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int nb_patches_h = max_nb_patches_h;
        int nb_patches_w = max_nb_patches_w;

        if (tgt_sizes.shape().size() == 2 && batch_idx < tgt_sizes.shape()[0]) {
          nb_patches_h = tgt_sizes.at<int32_t>({batch_idx, 0});
          nb_patches_w = tgt_sizes.at<int32_t>({batch_idx, 1});
        }

        // Create fractional coordinates like torch.arange(0, 1 - 1e-6, 1 / nb_patches_h/w)
        std::vector<float> fractional_coords_h;
        std::vector<float> fractional_coords_w;

        float step_h = 1.0f / static_cast<float>(nb_patches_h);
        float step_w = 1.0f / static_cast<float>(nb_patches_w);

        fractional_coords_h.reserve(nb_patches_h);
        for (int i = 0; i < nb_patches_h; ++i) { fractional_coords_h.push_back(i * step_h); }
        fractional_coords_w.reserve(nb_patches_w);
        for (int i = 0; i < nb_patches_w; ++i) { fractional_coords_w.push_back(i * step_w); }

        // Bucketize coordinates (equivalent to torch.bucketize with right=True)
        std::vector<int> bucket_coords_h(nb_patches_h);
        std::vector<int> bucket_coords_w(nb_patches_w);

        for (int h = 0; h < nb_patches_h; ++h) {
          float coord = fractional_coords_h[h];
          int bucket = 0;
          for (size_t i = 0; i < boundaries.size(); ++i) {
            if (coord < boundaries[i]) {
              bucket = static_cast<int>(i);
              break;
            }
            bucket = static_cast<int>(i + 1);
          }
          bucket_coords_h[h] = bucket;
        }

        for (int w = 0; w < nb_patches_w; ++w) {
          float coord = fractional_coords_w[w];
          int bucket = 0;
          for (size_t i = 0; i < boundaries.size(); ++i) {
            if (coord < boundaries[i]) {
              bucket = static_cast<int>(i);
              break;
            }
            bucket = static_cast<int>(i + 1);
          }
          bucket_coords_w[w] = bucket;
        }

        // Create pos_ids like Python: (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
        std::vector<int> pos_ids;
        for (int h = 0; h < nb_patches_h; ++h) {
          for (int w = 0; w < nb_patches_w; ++w) {
            int pos_id = bucket_coords_h[h] * num_patches_per_side_ + bucket_coords_w[w];
            pos_ids.push_back(pos_id);
          }
        }

        // Apply pos_ids only where patch_attention_mask is True (now it's 1D)
        // position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        int pos_ids_idx = 0;
        for (int flat_idx = 0; flat_idx < max_patches; ++flat_idx) {
          uint8_t mask_val = patch_attention_mask.at<uint8_t>({batch_idx, 0, flat_idx});
          if (mask_val && pos_ids_idx < pos_ids.size()) {
            position_ids.at<int64_t>({batch_idx, flat_idx}) = pos_ids[pos_ids_idx];
            pos_ids_idx++;
          }
        }
      }

      auto pos_embeddings = position_embedding_(position_ids);
      embeddings = embeddings + pos_embeddings;
    } else {
      auto seq_len = embeddings.shape()[1];
      auto position_ids = Tensor::arange(0, seq_len, kInt64).view({1, seq_len});
      auto pos_embeddings = position_embedding_(position_ids);
      embeddings = embeddings + pos_embeddings;
    }

    return {embeddings};
  }
};

// SigLIP MLP
class SiglipMLP final : public nn::Module {
  int32_t hidden_size_;
  int32_t intermediate_size_;

  nn::Linear fc1_;
  nn::Linear fc2_;
  nn::GELU activation_fn_;

 public:
  SiglipMLP() = default;

  inline SiglipMLP(const std::string& name, const MiniCPMOConfig& config) : nn::Module(name) {
    hidden_size_ = config.vision_hidden_size;
    intermediate_size_ = config.vision_intermediate_size;

    fc1_ = reg<nn::Linear>("fc1", hidden_size_, intermediate_size_, true);
    fc2_ = reg<nn::Linear>("fc2", intermediate_size_, hidden_size_, true);
    activation_fn_ = reg<nn::GELU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];

    auto x = fc1_(hidden_states);
    x = activation_fn_(x);
    x = fc2_(x);

    return {x};
  }
};

// SigLIP Multi-Head Attention
class SiglipAttention final : public nn::Module {
  int32_t embed_dim_;
  int32_t num_heads_;
  int32_t head_dim_;
  float scale_;

  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear q_proj_;
  nn::Linear out_proj_;

 public:
  SiglipAttention() = default;

  inline SiglipAttention(const std::string& name, const MiniCPMOConfig& config) : nn::Module(name) {
    embed_dim_ = config.vision_hidden_size;
    num_heads_ = config.vision_num_attention_heads;
    head_dim_ = embed_dim_ / num_heads_;
    scale_ = 1.0f / sqrtf(static_cast<float>(head_dim_));

    k_proj_ = reg<nn::Linear>("k_proj", embed_dim_, embed_dim_, true);
    v_proj_ = reg<nn::Linear>("v_proj", embed_dim_, embed_dim_, true);
    q_proj_ = reg<nn::Linear>("q_proj", embed_dim_, embed_dim_, true);
    out_proj_ = reg<nn::Linear>("out_proj", embed_dim_, embed_dim_, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto attention_mask = inputs.size() > 1 ? inputs[1] : Tensor::nil();

    auto batch_size = hidden_states.shape()[0];
    auto seq_len = hidden_states.shape()[1];

    // Apply projections
    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    // Reshape for multi-head attention: [B, seq_len, embed_dim] -> [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len,
    // head_dim]
    query_states = query_states.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
    key_states = key_states.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
    value_states = value_states.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);

    // Compute attention scores: [B, num_heads, seq_len, seq_len]
    auto attn_weights = nn::functional::matmul(query_states, key_states.transpose(-2, -1)) * scale_;

    // Apply attention mask if provided
    if (!attention_mask.isNil()) { attn_weights = attn_weights + attention_mask; }

    // Apply softmax
    attn_weights = nn::functional::softmax(attn_weights, -1);

    // Apply attention to values
    auto attn_output = nn::functional::matmul(attn_weights, value_states);

    // Reshape back: [B, num_heads, seq_len, head_dim] -> [B, seq_len, num_heads, head_dim] -> [B, seq_len, embed_dim]
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, embed_dim_});

    // Apply output projection
    auto output = out_proj_(attn_output);

    return {output};
  }
};

// SigLIP Encoder Layer
class SiglipEncoderLayer final : public nn::Module {
  int32_t embed_dim_;

  SiglipAttention self_attn_;
  nn::LayerNorm layer_norm1_;
  SiglipMLP mlp_;
  nn::LayerNorm layer_norm2_;

 public:
  SiglipEncoderLayer() = default;

  inline SiglipEncoderLayer(const std::string& name, const MiniCPMOConfig& config) : nn::Module(name) {
    embed_dim_ = config.vision_hidden_size;

    self_attn_ = reg<SiglipAttention>("self_attn", config);
    layer_norm1_ = reg<nn::LayerNorm>("layer_norm1", std::vector<int32_t>{embed_dim_}, true, true, 1e-6);
    mlp_ = reg<SiglipMLP>("mlp", config);
    layer_norm2_ = reg<nn::LayerNorm>("layer_norm2", std::vector<int32_t>{embed_dim_}, true, true, 1e-6);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto attention_mask = inputs.size() > 1 ? inputs[1] : Tensor::nil();

    // TODO: Perf Issue
    // attention > 800ms (1k tokens)
    // mlp > 600ms

    // Self attention with residual connection
    auto residual = hidden_states;
    auto normed = layer_norm1_(hidden_states);
    auto attn_output = self_attn_(normed, attention_mask)[0];
    auto after_attn = residual + attn_output;

    // MLP with residual connection
    residual = after_attn;
    normed = layer_norm2_(after_attn);
    auto mlp_output = mlp_(normed)[0];
    auto output = residual + mlp_output;

    return {output};
  }
};

// SigLIP Vision Encoder
class SiglipVisionEncoder final : public nn::Module {
  int32_t num_layers_;
  std::vector<SiglipEncoderLayer> layers_;

 public:
  SiglipVisionEncoder() = default;

  inline SiglipVisionEncoder(const std::string& name, const MiniCPMOConfig& config) : nn::Module(name) {
    num_layers_ = config.vision_num_hidden_layers;

    for (int i = 0; i < num_layers_; ++i) { layers_.push_back(reg<SiglipEncoderLayer>("layers." + std::to_string(i), config)); }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    const auto& inputs_embeds = inputs[0];
    auto attention_mask = inputs.size() > 1 ? inputs[1] : Tensor::nil();

    auto hidden_states = inputs_embeds;
    for (auto& layer : layers_) { hidden_states = layer(hidden_states, attention_mask)[0]; }
    return {hidden_states};
  }
};

// Main SigLIP Vision Model
class SiglipVisionModel final : public nn::Module {
  int32_t embed_dim_;
  int32_t patch_size_;  // Add patch_size_ to access in forward

  SiglipVisionEmbeddings embeddings_;
  SiglipVisionEncoder encoder_;
  nn::LayerNorm post_layernorm_;

 public:
  SiglipVisionModel() = default;

  inline SiglipVisionModel(const std::string& name, const MiniCPMOConfig& config) : nn::Module(name) {
    embed_dim_ = config.vision_hidden_size;
    patch_size_ = config.vision_patch_size;  // Initialize patch_size_

    embeddings_ = reg<SiglipVisionEmbeddings>("embeddings", config);
    encoder_ = reg<SiglipVisionEncoder>("encoder", config);
    post_layernorm_ = reg<nn::LayerNorm>("post_layernorm", std::vector<int32_t>{embed_dim_}, true, true, 1e-6);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto pixel_values = inputs[0];
    auto tgt_sizes = inputs.size() > 1 ? inputs[1] : Tensor::nil();

    auto batch_size = pixel_values.shape()[0];
    int max_patches = 0;
    // Calculate max_patches based on tgt_sizes
    for (int i = 0; i < tgt_sizes.shape()[0]; i++) {
      if (tgt_sizes.at<int32_t>({i, 0}) > 0 && tgt_sizes.at<int32_t>({i, 1}) > 0) {
        int patches = (tgt_sizes.at<int32_t>({i, 0})) * (tgt_sizes.at<int32_t>({i, 1}));
        if (patches > max_patches) max_patches = patches;
      }
    }
    auto patch_attention_mask = Tensor::empty({batch_size, 1, max_patches}, kUInt8).alloc();
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < max_patches; j++) { patch_attention_mask.at<uint8_t>({i, 0, j}) = 0; }
      if (!tgt_sizes.isNil() && i < tgt_sizes.shape()[0]) {
        int nb_patches_h = tgt_sizes.at<int32_t>({i, 0});
        int nb_patches_w = tgt_sizes.at<int32_t>({i, 1});
        int valid_patches = nb_patches_h * nb_patches_w;
        for (int j = 0; j < valid_patches && j < max_patches; j++) { patch_attention_mask.at<uint8_t>({i, 0, j}) = 1; }
      }
    }
    std::vector<Tensor> hidden_states_result;
    if (tgt_sizes.isNil()) {
      hidden_states_result = embeddings_(pixel_values, Tensor::nil(), patch_attention_mask);
    } else {
      hidden_states_result = embeddings_(pixel_values, tgt_sizes, patch_attention_mask);
    }
    auto hidden_states = hidden_states_result[0];  // [B, num_patches, embed_dim]

    patch_attention_mask = patch_attention_mask.squeeze(1);  // [B, max_patches]

    // Create attention mask for encoder (4D mask for multi-head attention)
    // TODO: this will take about 100ms, optimize it
    Tensor attention_mask = Tensor::nil();
    if (!patch_attention_mask.isNil()) {
      auto batch_size = patch_attention_mask.shape()[0];
      auto max_patches = patch_attention_mask.shape()[1];

      bool all_valid = true;
      for (int i = 0; i < batch_size && all_valid; i++) {
        for (int j = 0; j < max_patches && all_valid; j++) {
          uint8_t mask_val = patch_attention_mask.at<uint8_t>({i, j});
          if (mask_val == 0) { all_valid = false; }
        }
      }
      if (!all_valid) {
        // Convert patch_attention_mask to float and create 4D attention mask
        auto patch_mask_float = Tensor::empty({batch_size, max_patches}, kFloat32).alloc();
        for (int i = 0; i < batch_size; i++) {
          for (int j = 0; j < max_patches; j++) {
            uint8_t mask_val = patch_attention_mask.at<uint8_t>({i, j});
            patch_mask_float.at<float>({i, j}) = mask_val ? 1.0f : 0.0f;
          }
        }

        // Create 4D attention mask: [B, 1, max_patches, max_patches]
        attention_mask = Tensor::empty({batch_size, 1, max_patches, max_patches}, kFloat32).alloc();

        // Optimize with cache-friendly access patterns and reduced redundant accesses
        for (int b = 0; b < batch_size; b++) {
          // Pre-fetch mask values for this batch to improve cache locality
          std::vector<float> batch_mask(max_patches);
          for (int p = 0; p < max_patches; p++) { batch_mask[p] = patch_mask_float.at<float>({b, p}); }

          // Compute attention mask for this batch with optimized memory access
          for (int i = 0; i < max_patches; i++) {
            float mask_i = batch_mask[i];
            // Process row in chunks for better cache utilization
            for (int j = 0; j < max_patches; j++) {
              float mask_j = batch_mask[j];
              // Both positions must be valid (branchless computation)
              float final_mask = (mask_i > 0.0f && mask_j > 0.0f) ? 0.0f : -1e9f;
              attention_mask.at<float>({b, 0, i, j}) = final_mask;
            }
          }
        }
      }
    }

    auto encoder_outputs = encoder_(hidden_states, attention_mask)[0];

    // Apply post layer norm
    auto last_hidden_state = post_layernorm_(encoder_outputs);

    return {last_hidden_state};
  }
};

}  // namespace mllm::models::minicpmo
