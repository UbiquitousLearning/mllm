// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <optional>
#include <algorithm>

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/layers/Param.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/deepseek_ocr/configuration_deepseek_ocr.hpp"

namespace mllm::models::deepseek_ocr {

//===----------------------------------------------------------------------===//
// MLP Projector For Mapping Visual Tokens to Text Token Space
//===----------------------------------------------------------------------===//
class MlpProjector final : public nn::Module {
  nn::Linear layers_;

 public:
  MlpProjector() = default;

  MlpProjector(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    layers_ = reg<nn::Linear>("layers", 2048, 1280, true, config.mlp_projector_linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {layers_(inputs[0])};
  }
};

//===----------------------------------------------------------------------===//
// CLIP
//
// CLIP params is hard coded. Just like what deepseek official model does.
//
// vit_model_cfg = adict(
//     num_layers=24,
//     hidden_size=1024,
//     num_heads = 16,
//     num_attention_heads=16,
//     ffn_hidden_size=4096,
//     seq_length=256,
//     max_position_embeddings=256,
//     use_flash_attn=False,
//     understand_projector_stride=2,
//     hidden_dropout = 0.0,
//     attention_dropout = 0.0,
//     no_persist_layer_norm = False,
//     layernorm_epsilon = 1e-5,
//     pre_layernorm_epsilon = 1e-5,
//     image_size = 224,
//     patch_size = 14,
//     recompute_list = []
// )

// def build_clip_l():
//     return VitModel(
//         cfg=vit_model_cfg,
//         freeze_embed=False,
//         freeze_pre_norm=False,
//     )
//===----------------------------------------------------------------------===//
class CLIPVisionEmbeddings final : public nn::Module {
  int embed_dim_;
  int image_size_;
  int patch_size_;
  nn::Param class_embedding_;
  nn::Conv2D patch_embedding_;
  int num_patches_;
  int num_positions_;
  nn::Embedding position_embedding_;

 public:
  CLIPVisionEmbeddings() = default;

  CLIPVisionEmbeddings(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    embed_dim_ = 1024;
    image_size_ = 224;
    patch_size_ = 14;
    num_patches_ = (image_size_ / patch_size_) * (image_size_ / patch_size_);
    num_positions_ = num_patches_ + 1;

    // [embed_dim], aka [1024]
    class_embedding_ = reg<nn::Param>("class_embedding", getModuleName() + ".class_embedding");
    patch_embedding_ = reg<nn::Conv2D>("patch_embedding", 3, embed_dim_, Tensor::shape_t{14, 14}, Tensor::shape_t{14, 14},
                                       Tensor::shape_t{0, 0}, Tensor::shape_t{1, 1}, false);
    position_embedding_ = reg<nn::Embedding>("position_embedding", num_positions_, embed_dim_);

    // Register a buffer
    registerBuffer("position_ids", Tensor::arange(0, num_positions_, 1, kInt64, kCPU).view({1, -1}));
  }

  Tensor getAbsPos(Tensor abs_pos, int32_t tgt_size) {
    // abs_pos : L, C
    // tgt_size : M
    // return : M, C

    auto dim = abs_pos.size(-1);
    auto abs_pos_new = abs_pos.squeeze(0);
    auto cls_token = abs_pos_new[{{kAll, 1}, kAll}].contiguous();
    auto old_pos_embed = abs_pos_new[{{1, kAll}, kAll}].contiguous();

    auto src_size = int(std::sqrt(abs_pos_new.shape()[0] - 1));
    tgt_size = int(std::sqrt(tgt_size));
    auto dtype = abs_pos.dtype();

    if (src_size != tgt_size) {
      old_pos_embed = old_pos_embed.view({1, src_size, src_size, dim}).permute({0, 3, 1, 2});
      old_pos_embed = old_pos_embed.to(kFloat32);
      auto new_pos_embed = nn::functional::interpolateBySize(old_pos_embed, {tgt_size, tgt_size},
                                                             aops::InterpolateOpMode::kBicubic, false, true);
      new_pos_embed = new_pos_embed.permute({0, 2, 3, 1});
      new_pos_embed = new_pos_embed.view({tgt_size * tgt_size, dim});
      auto vision_pos_embed = nn::functional::concat({cls_token, new_pos_embed}, 0);
      vision_pos_embed = vision_pos_embed.view({1, tgt_size * tgt_size + 1, dim});
      return vision_pos_embed;
    } else {
      return abs_pos;
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto pixel_values = Tensor::nil();
    auto patch_embeds = Tensor::nil();

    if (inputs.size() == 1) {
      pixel_values = inputs[0];
    } else if (inputs.size() == 2) {
      pixel_values = inputs[0];
      patch_embeds = inputs[1];
    }

    auto batch_size = pixel_values.shape()[0];

    if (!patch_embeds) { patch_embeds = patch_embedding_(pixel_values); }

    // Flatten and transpose.
    // patch_embeds original shape is [batch, out_channel, width, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2);  // [batch, width * grid * grid, out_channel]

    // [batch, 1, 1024]
    // Same as expand(batch_size, 1, -1)
    auto class_embeds = class_embedding_.weight().view({1, 1, -1}).repeat(batch_size, 0);

    auto embeddings = nn::functional::concat({class_embeds, patch_embeds}, 1);
    embeddings = embeddings + getAbsPos(position_embedding_(getBuffer("position_ids")), embeddings.size(1));

    return {embeddings};
  }
};

class NoTPFeedForward final : public nn::Module {
  nn::Linear fc1_;
  nn::Linear fc2_;
  nn::QuickGELU act_;

 public:
  NoTPFeedForward() = default;

  NoTPFeedForward(const std::string& name, int32_t dim, int32_t hidden_dim, const DpskOcrConfig& config) : nn::Module(name) {
    fc1_ = reg<nn::Linear>("fc1", dim, hidden_dim, true, config.clip_linear_impl_type);
    fc2_ = reg<nn::Linear>("fc2", hidden_dim, dim, true, config.clip_linear_impl_type);
    act_ = reg<nn::QuickGELU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {fc2_(act_(fc1_(inputs[0])))};
  }
};

class NoTPAttention final : public nn::Module {
  int num_heads_;
  int n_local_heads_;
  int head_dim_;
  int max_seq_len_;
  nn::Linear qkv_proj_;
  nn::Linear out_proj_;

 public:
  NoTPAttention() = default;

  NoTPAttention(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    num_heads_ = 16;
    n_local_heads_ = 16;
    head_dim_ = 1024 / 16;
    max_seq_len_ = 256;

    qkv_proj_ = reg<nn::Linear>("qkv_proj", 1024, 1024 * 3, true, config.clip_linear_impl_type);
    out_proj_ = reg<nn::Linear>("out_proj", 1024, 1024, true, config.clip_linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& x = inputs[0];
    auto bsz = x.size(0);
    auto seqlen = x.size(1);

    auto xqkv = qkv_proj_(x);
    xqkv = xqkv.view({bsz, seqlen, 3, num_heads_, head_dim_});
    auto [xq, xk, xv] = nn::functional::split<3>(xqkv, 2);

    // FIXME: contiguous is not needed, actually.
    xq = xq.contiguous().squeeze(2);
    xk = xk.contiguous().squeeze(2);
    xv = xv.contiguous().squeeze(2);

    xq = xq.permute({0, 2, 1, 3});
    xk = xk.permute({0, 2, 1, 3});
    xv = xv.permute({0, 2, 1, 3});

    auto output = nn::functional::scaledDotProductAttention(xq, xk, xv);
    output = output.permute({0, 2, 1, 3}).view({bsz, seqlen, -1});
    output = out_proj_(output);
    return {output};
  }
};

class NoTPTransformerBlock final : public nn::Module {
  int n_heads_;
  int dim_;
  int head_dim_;
  NoTPAttention self_attn_;
  NoTPFeedForward mlp_;
  nn::LayerNorm layer_norm1_;
  nn::LayerNorm layer_norm2_;

 public:
  int layer_id_;

  NoTPTransformerBlock() = default;

  NoTPTransformerBlock(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    n_heads_ = 16;
    dim_ = 1024;
    head_dim_ = 1024 / 16;
    self_attn_ = reg<NoTPAttention>("self_attn", config);
    mlp_ = reg<NoTPFeedForward>("mlp", 1024, 4096, config);
    layer_norm1_ = reg<nn::LayerNorm>("layer_norm1", Tensor::shape_t{1024}, true, true, 1e-5);
    layer_norm2_ = reg<nn::LayerNorm>("layer_norm2", Tensor::shape_t{1024}, true, true, 1e-5);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto residual = self_attn_(layer_norm1_(x))[0];
    auto h = x + residual;
    auto out = h + mlp_(layer_norm2_(h))[0];
    return {out};
  }
};

class NoTPTransformer final : public nn::Module {
  int num_layers_;
  nn::ModuleList<NoTPTransformerBlock> layers_;

 public:
  NoTPTransformer() = default;

  NoTPTransformer(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    num_layers_ = 24;
    layers_ = reg<nn::ModuleList<NoTPTransformerBlock>>("layers", num_layers_, config);
    for (auto [idx, layer] : enumerate(layers_.list())) { layer.layer_id_ = idx; }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    for (auto [idx, layer] : enumerate(layers_.list())) { hidden_states = layer(hidden_states)[0]; }

    return {hidden_states};
  }
};

class VitModel final : public nn::Module {
  CLIPVisionEmbeddings embeddings_;
  NoTPTransformer transformer_;
  nn::LayerNorm pre_layernorm_;  ///< input must in fp32 dtype.

 public:
  VitModel() = default;

  VitModel(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    embeddings_ = reg<CLIPVisionEmbeddings>("embeddings", config);
    transformer_ = reg<NoTPTransformer>("transformer", config);

    // NOTE:
    // Yes!!!, Its pre_layrnorm! Deepseek Typo!.
    pre_layernorm_ = reg<nn::LayerNorm>("pre_layrnorm", Tensor::shape_t{1024}, true, true, 1e-5);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto patch_embeds = inputs[1];

    auto output = embeddings_(x, patch_embeds)[0];
    output = pre_layernorm_(output);
    output = transformer_(output)[0];
    return {output};
  }
};

//===----------------------------------------------------------------------===//
// SAM
//===----------------------------------------------------------------------===//
class PatchEmbed final : public nn::Module {
  nn::Conv2D proj_;

 public:
  PatchEmbed() = default;

  PatchEmbed(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    proj_ = reg<nn::Conv2D>("proj", 3, 768, Tensor::shape_t{16, 16}, Tensor::shape_t{16, 16}, Tensor::shape_t{0, 0},
                            Tensor::shape_t{1, 1}, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    // B C H W -> B H W C
    return {proj_(x).permute({0, 2, 3, 1})};
  }
};

class MLPBlock final : public nn::Module {
  nn::Linear lin1_;
  nn::Linear lin2_;
  nn::GELU act_;

 public:
  MLPBlock() = default;

  MLPBlock(const std::string& name, int embedding_dim, int mlp_dim, const DpskOcrConfig& config) : nn::Module(name) {
    lin1_ = reg<nn::Linear>("lin1", embedding_dim, mlp_dim, true, config.sam_linear_impl_type);
    lin2_ = reg<nn::Linear>("lin2", mlp_dim, embedding_dim, true, config.sam_linear_impl_type);
    act_ = reg<nn::GELU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {lin2_(act_(lin1_(inputs[0])))};
  }
};

class Attention final : public nn::Module {
  int num_heads_;
  bool use_rel_pos_;

  nn::Linear qkv_;
  nn::Linear proj_;
  nn::Param rel_pos_h_;
  nn::Param rel_pos_w_;

 public:
  int layer_idx_;

  Attention() = default;

  Attention(const std::string& name, int dim, int num_heads, bool qkv_bias, bool use_rel_pos,
            std::optional<std::tuple<int, int>> input_size, const DpskOcrConfig& config)
      : nn::Module(name) {
    num_heads_ = num_heads;
    use_rel_pos_ = use_rel_pos;

    qkv_ = reg<nn::Linear>("qkv", dim, dim * 3, qkv_bias, config.sam_linear_impl_type);
    proj_ = reg<nn::Linear>("proj", dim, dim, true, config.sam_linear_impl_type);
    if (use_rel_pos) {
      rel_pos_h_ = reg<nn::Param>("rel_pos_h", getModuleName() + ".rel_pos_h");
      rel_pos_w_ = reg<nn::Param>("rel_pos_w", getModuleName() + ".rel_pos_w");
    }
  }

  // Get relative positional embeddings according to the relative positions of query and key sizes.
  Tensor getRelPos(int q_size, int k_size, const Tensor& rel_pos_) {
    auto rel_pos = rel_pos_;
    auto max_rel_dist = 2 * std::max(q_size, k_size) - 1;
    Tensor rel_pos_resized = Tensor::nil();

    if (rel_pos.size(0) != max_rel_dist) {
      auto dtype = rel_pos.dtype();
      rel_pos = rel_pos.to(kFloat32);
      rel_pos_resized = nn::functional::interpolateBySize(rel_pos.view({1, rel_pos.size(0), -1}).permute({0, 2, 1}),
                                                          {max_rel_dist}, aops::InterpolateOpMode::kLinear)
                            .to(dtype);
      rel_pos_resized = rel_pos_resized.view({-1, max_rel_dist}).permute({1, 0});
    } else {
      rel_pos_resized = rel_pos;
    }

    std::vector<float> q_coords(q_size);
    std::vector<float> k_coords(k_size);

    float q_scale = std::max((float)k_size / q_size, 1.0f);
    float k_scale = std::max((float)q_size / k_size, 1.0f);

    for (int i = 0; i < q_size; ++i) { q_coords[i] = i * q_scale; }
    for (int i = 0; i < k_size; ++i) { k_coords[i] = i * k_scale; }

    float offset = (k_size - 1) * k_scale;
    int embedding_dim = rel_pos_resized.size(1);
    auto out = Tensor::empty({q_size, k_size, embedding_dim}).alloc();

    for (int i = 0; i < q_size; ++i) {
      for (int j = 0; j < k_size; ++j) {
        float relative_coord_float = (q_coords[i] - k_coords[j]) + offset;
        int64_t relative_coord_long = static_cast<int64_t>(std::round(relative_coord_float));

        if (relative_coord_long < 0) relative_coord_long = 0;
        if (relative_coord_long >= max_rel_dist) relative_coord_long = max_rel_dist - 1;

        for (int d = 0; d < embedding_dim; ++d) {
          out.ptr<mllm_fp32_t>()[i * k_size * embedding_dim + j * embedding_dim + d] =
              *rel_pos_resized.offsettedPtr<mllm_fp32_t>({(int32_t)relative_coord_long, d});
        }
      }
    }

    return out;
  }

  // Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
  // https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
  std::tuple<Tensor, Tensor> addDecomposedRelPos(Tensor q, const Tensor& rel_pos_h, const Tensor& rel_pos_w,
                                                 std::tuple<int, int> q_size, std::tuple<int, int> k_size) {
    auto [q_h, q_w] = q_size;
    auto [k_h, k_w] = k_size;

    auto Rh = getRelPos(q_h, k_h, rel_pos_h);
    auto Rw = getRelPos(q_w, k_w, rel_pos_w);

    auto B = q.size(0);
    auto dim = q.size(2);

    auto r_q = q.view({B, q_h, q_w, dim});

    // Einsum
    // 1. bhwc,hkc->bhwk
    auto rel_h = Tensor::empty({B, q_h, q_w, k_h}).alloc();
    {
      auto* r_q_ptr = r_q.ptr<mllm_fp32_t>();
      auto* Rh_ptr = Rh.ptr<mllm_fp32_t>();
      auto* rel_h_ptr = rel_h.ptr<mllm_fp32_t>();
      // rel_h[b, h, w, k] = sum over c from 0 to dim-1 ( r_q[b, h, w, c] * Rh[h, k, c] )
      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < q_h; ++h) {
          for (int w = 0; w < q_w; ++w) {
            for (int k = 0; k < k_h; ++k) {
              float sum = 0.0f;
              const auto* p_r_q = r_q_ptr + (b * q_h * q_w * dim) + (h * q_w * dim) + (w * dim);
              const auto* p_Rh = Rh_ptr + (h * k_h * dim) + (k * dim);
              for (int c = 0; c < dim; ++c) { sum += p_r_q[c] * p_Rh[c]; }
              rel_h_ptr[(b * q_h * q_w * k_h) + (h * q_w * k_h) + (w * k_h) + k] = sum;
            }
          }
        }
      }
    }
    // 2. bhwc,wkc->bhwk
    auto rel_w = Tensor::empty({B, q_h, q_w, k_w}).alloc();
    {
      auto* r_q_ptr = r_q.ptr<mllm_fp32_t>();
      auto* Rw_ptr = Rw.ptr<mllm_fp32_t>();
      auto* rel_w_ptr = rel_w.ptr<mllm_fp32_t>();

      // rel_w[b, h, w, k] = sum over c from 0 to dim-1 ( r_q[b, h, w, c] * Rw[w, k, c] )
      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < q_h; ++h) {
          for (int w = 0; w < q_w; ++w) {
            for (int k = 0; k < k_w; ++k) {
              float sum = 0.0f;
              const auto* p_r_q = r_q_ptr + (b * q_h * q_w * dim) + (h * q_w * dim) + (w * dim);
              const auto* p_Rw = Rw_ptr + (w * k_w * dim) + (k * dim);
              for (int c = 0; c < dim; ++c) { sum += p_r_q[c] * p_Rw[c]; }
              rel_w_ptr[(b * q_h * q_w * k_w) + (h * q_w * k_w) + (w * k_w) + k] = sum;
            }
          }
        }
      }
    }

    rel_h = rel_h.unsqueeze(-1);
    rel_w = rel_w.unsqueeze(-2);
    rel_h = rel_h.view({B, q_h * q_w, k_h, 1});
    rel_w = rel_w.view({B, q_h * q_w, 1, k_w});
    return {rel_h, rel_w};
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto B = x.size(0);
    auto H = x.size(1);
    auto W = x.size(2);

    // qkv with shape (3, B, nHead, H * W, C)
    auto qkv = qkv_(x)
                   .view({
                       B,
                       H * W,
                       3,
                       num_heads_,
                       -1,
                   })
                   .permute({2, 0, 3, 1, 4});
    qkv = qkv.view({3, B * num_heads_, H * W, -1});
    auto [q, k, v] = nn::functional::split<3>(qkv, 0);

    q = q.view({B * num_heads_, H * W, -1});
    k = k.view({B * num_heads_, H * W, -1});
    v = v.view({B * num_heads_, H * W, -1});

    auto rel_h = Tensor::nil();
    auto rel_w = Tensor::nil();

    if (use_rel_pos_) {
      std::tie(rel_h, rel_w) = addDecomposedRelPos(q, rel_pos_h_.weight(), rel_pos_w_.weight(), {H, W}, {H, W});
    }

    q = q.view({B, num_heads_, H * W, -1});
    k = k.view({B, num_heads_, H * W, -1});
    v = v.view({B, num_heads_, H * W, -1});

    if (use_rel_pos_) {
      rel_h = rel_h.view({B, num_heads_, rel_h.size(1), rel_h.size(2), rel_h.size(3)});
      rel_w = rel_w.view({B, num_heads_, rel_w.size(1), rel_w.size(2), rel_w.size(3)});

      // Dual broadcast is not supported in cpu backend. So we need to repeat rel_h and rel_w
      // torch.Size([54, 12, 196, 14, 1])
      // torch.Size([54, 12, 196, 1, 14])
      auto _dim_neg_1 = rel_w.size(4);
      auto _dim_neg_2 = rel_h.size(3);
      rel_h = rel_h.repeat(_dim_neg_1, -1);
      rel_w = rel_w.repeat(_dim_neg_2, -2);
      MLLM_RT_ASSERT_EQ(rel_h.shape(), rel_w.shape());
      auto attn_bias = (rel_h + rel_w).view({B, num_heads_, rel_h.size(2), rel_h.size(3) * rel_w.size(4)});
      x = nn::functional::scaledDotProductAttention(q, k, v, attn_bias);
    } else {
      x = nn::functional::scaledDotProductAttention(q, k, v);
    }

    x = x.view({B, num_heads_, H, W, -1}).permute({0, 2, 3, 1, 4}).view({B, H, W, -1});
    x = proj_(x);
    return {x};
  }
};

class Block final : public nn::Module {
  nn::LayerNorm norm1_;
  nn::LayerNorm norm2_;
  MLPBlock mlp_;
  int window_size_;

 public:
  Attention attn_;
  int layer_idx_;

  Block() = default;

  Block(const std::string& name, int dim, int num_heads, float mlp_ratio, bool qkv_bias, bool use_rel_pos, int window_size,
        std::optional<std::tuple<int, int>> input_size, const DpskOcrConfig& config)
      : nn::Module(name) {
    norm1_ = reg<nn::LayerNorm>("norm1", Tensor::shape_t{dim});
    attn_ =
        reg<Attention>("attn", dim, num_heads, qkv_bias, use_rel_pos,
                       window_size == 0 ? input_size : std::make_optional(std::make_tuple(window_size, window_size)), config);
    norm2_ = reg<nn::LayerNorm>("norm2", Tensor::shape_t{dim});
    mlp_ = reg<MLPBlock>("mlp", dim, (int)(dim * mlp_ratio), config);
    window_size_ = window_size;
  }

  std::tuple<Tensor, std::tuple<int, int>> windowPartition(Tensor x, int window_size) {
    auto B = x.size(0);
    auto H = x.size(1);
    auto W = x.size(2);
    auto C = x.size(3);

    auto pad_h = (window_size - H % window_size) % window_size;
    auto pad_w = (window_size - W % window_size) % window_size;

    if (pad_h > 0 || pad_w > 0) { x = nn::functional::pad(x, {0, 0, 0, pad_w, 0, pad_h}); }

    auto Hp = H + pad_h;
    auto Wp = W + pad_w;

    x = x.view({B, Hp / window_size, window_size, Wp / window_size, window_size, C});
    auto window = x.permute({0, 1, 3, 2, 4, 5}).view({-1, window_size, window_size, C});
    return {window, {Hp, Wp}};
  }

  Tensor windowUnpartition(Tensor windows, int window_size, std::tuple<int, int> pad_wh, std::tuple<int, int> hw) {
    auto [Hp, Wp] = pad_wh;
    auto [H, W] = hw;
    auto B = windows.size(0) / (Hp * Wp / window_size / window_size);
    auto x = windows.view({B, Hp / window_size, Wp / window_size, window_size, window_size, -1});
    x = x.permute({0, 1, 3, 2, 4, 5}).view({B, Hp, Wp, -1});

    if (Hp > H || Wp > W) { x = x[{kAll, {kAll, H}, {kAll, W}, kAll}].contiguous(); }

    return x;
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto shortcut = x;
    x = norm1_(x);

    // Window partition
    int H = 0;
    int W = 0;
    std::tuple<int, int> pad_hw;
    if (window_size_ > 0) {
      H = x.size(1);
      W = x.size(2);
      std::tie(x, pad_hw) = windowPartition(x, window_size_);
    }

    x = attn_(x)[0];

    // Reverse window partition
    if (window_size_ > 0) { x = windowUnpartition(x, window_size_, pad_hw, {H, W}); }

    x = shortcut + x;
    x = x + mlp_(norm2_(x))[0];

    return {x};
  }
};

class Blocks final : public nn::Module {
  std::vector<Block> blocks_;

 public:
  Blocks() = default;

  Blocks(const std::string& name, int nums, const std::vector<int>& global_attn_indexes, const DpskOcrConfig& config)
      : nn::Module(name) {
    for (int i = 0; i < nums; ++i) {
      bool is_in = std::find(global_attn_indexes.begin(), global_attn_indexes.end(), i) == global_attn_indexes.end();
      auto this_block_window_size = is_in ? 14 : 0;
      blocks_.emplace_back(reg<Block>(std::to_string(i), 768, 12, 4.0, true, true, this_block_window_size,
                                      std::make_optional(std::make_tuple(1024 / 16, 1024 / 16)), config));
      blocks_[i].layer_idx_ = i;
      blocks_[i].attn_.layer_idx_ = i;
    }
  };

  std::vector<Block>& list() { return blocks_; }
};

class ImageEncoderViT final : public nn::Module {
  PatchEmbed patch_embed_;
  nn::Param pos_embed_;
  Blocks blocks_;
  nn::Sequential neck_;
  nn::Conv2D net_2_;
  nn::Conv2D net_3_;

 public:
  ImageEncoderViT() = default;

  ImageEncoderViT(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    patch_embed_ = reg<PatchEmbed>("patch_embed", config);
    pos_embed_ = reg<nn::Param>("pos_embed", getModuleName() + ".pos_embed");

    // block_nums = 12
    // embed_dim = 768
    // num_heads = 12
    // mlp_ratio = 4.f
    // qkv_bias = true
    // use_rel_pos = true
    // window_size = 14
    blocks_ = reg<Blocks>("blocks", 12, std::vector<int>{2, 5, 8, 11}, config);

    neck_ = reg<nn::Sequential>("neck")
                .add<nn::Conv2D>(768, 256, Tensor::shape_t{1, 1}, Tensor::shape_t{1, 1}, Tensor::shape_t{0, 0},
                                 Tensor::shape_t{1, 1}, false)
                .add<nn::LayerNorm2D>(256)
                .add<nn::Conv2D>(256, 256, Tensor::shape_t{3, 3}, Tensor::shape_t{1, 1}, Tensor::shape_t{1, 1},
                                 Tensor::shape_t{1, 1}, false)
                .add<nn::LayerNorm2D>(256);

    net_2_ = reg<nn::Conv2D>("net_2", 256, 512, Tensor::shape_t{3, 3}, Tensor::shape_t{2, 2}, Tensor::shape_t{1, 1},
                             Tensor::shape_t{1, 1}, false);
    net_3_ = reg<nn::Conv2D>("net_3", 512, 1024, Tensor::shape_t{3, 3}, Tensor::shape_t{2, 2}, Tensor::shape_t{1, 1},
                             Tensor::shape_t{1, 1}, false);
  }

  Tensor getAbsPosSam(Tensor abs_pos, int tgt_size) {
    auto dtype = abs_pos.dtype();
    auto src_size = abs_pos.size(1);

    if (src_size != tgt_size) {
      auto old_pos_embed = abs_pos.permute({0, 3, 1, 2});
      old_pos_embed = old_pos_embed.to(kFloat32);
      // clang-format off
      auto new_pos_embed = nn::functional::interpolateBySize(old_pos_embed, {tgt_size, tgt_size}, aops::InterpolateOpMode::kBicubic, false, true).to(dtype);
      // clang-format on
      new_pos_embed = new_pos_embed.permute({0, 2, 3, 1});
      return new_pos_embed;
    } else {
      return abs_pos;
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    x = patch_embed_(x)[0];
    x = x + getAbsPosSam(pos_embed_.weight(), x.size(1));
    for (auto& blk : blocks_.list()) { x = blk(x)[0]; }

    x = neck_(x.permute({0, 3, 1, 2}))[0];
    x = net_2_(x);
    x = net_3_(x);
    return {x};
  }
};

}  // namespace mllm::models::deepseek_ocr
