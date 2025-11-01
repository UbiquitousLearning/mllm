// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/minicpm_o2_6/modeling_vector_quantize.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/layers/LayerNorm.hpp"
#include "mllm/nn/layers/Linear.hpp"
#include "mllm/nn/layers/Param.hpp"
#include "mllm/nn/Functional.hpp"
#include <cstdint>
#include <string>
#include <cmath>

namespace mllm::models::minicpmo {

inline Tensor get2DSinCosPosEmbed(int32_t embed_dim, const std::vector<int32_t>& image_size) {
  int32_t grid_h_size = (image_size.size() == 1) ? image_size[0] : image_size[0];
  int32_t grid_w_size = (image_size.size() == 1) ? image_size[0] : image_size[1];

  Tensor pos_embed = Tensor::empty({grid_h_size, grid_w_size, embed_dim}, kFloat32).alloc();

  int32_t half_dim = embed_dim / 2;

  for (int32_t h = 0; h < grid_h_size; ++h) {
    for (int32_t w = 0; w < grid_w_size; ++w) {
      for (int32_t i = 0; i < half_dim / 2; ++i) {
        float omega = 1.0f / std::pow(10000.0f, 2.0f * i / half_dim);
        *pos_embed.offsettedPtr<float>({h, w, i}) = std::sin(w * omega);
        *pos_embed.offsettedPtr<float>({h, w, i + half_dim / 2}) = std::cos(w * omega);
      }

      for (int32_t i = 0; i < half_dim / 2; ++i) {
        float omega = 1.0f / std::pow(10000.0f, 2.0f * i / half_dim);
        *pos_embed.offsettedPtr<float>({h, w, half_dim + i}) = std::sin(h * omega);
        *pos_embed.offsettedPtr<float>({h, w, half_dim + i + half_dim / 2}) = std::cos(h * omega);
      }
    }
  }

  return pos_embed;
}

class ResamplerAttention : public nn::Module {
  int32_t embed_dim_;
  int32_t num_heads_;
  int32_t head_dim_;
  nn::Param in_proj_weight_;
  nn::Param in_proj_bias_;
  nn::Linear out_proj_;

 public:
  ResamplerAttention() = default;

  ResamplerAttention(const std::string& name, int32_t embed_dim, int32_t num_heads)
      : nn::Module(name), embed_dim_(embed_dim), num_heads_(num_heads) {
    head_dim_ = embed_dim_ / num_heads_;

    // in_proj_weight [3*embed_dim, embed_dim]
    in_proj_weight_ =
        reg<nn::Param>("in_proj_weight", getModuleName() + ".in_proj.weight", Tensor::shape_t{3 * embed_dim_, embed_dim_});
    in_proj_bias_ = reg<nn::Param>("in_proj_bias", getModuleName() + ".in_proj.bias", Tensor::shape_t{3 * embed_dim_});
    out_proj_ = reg<nn::Linear>("out_proj", embed_dim_, embed_dim_, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    const auto& query = inputs[0];  // [num_queries, embed_dim]
    const auto& key = inputs[1];    // [seq_len, embed_dim]
    const auto& value = inputs[2];  // [seq_len, embed_dim]

    auto key_padding_mask = Tensor();
    bool has_key_padding_mask = false;
    if (inputs.size() > 3) {
      key_padding_mask = inputs[3];  // [seq_len] optional
      has_key_padding_mask = true;
    }

    auto num_queries = query.shape()[0];
    auto seq_len = key.shape()[0];

    // Perform packed in-projection: [query|key|value] = input @ in_proj_weight.T + in_proj_bias
    // For cross-attention: q comes from query, k,v come from key_value
    auto q_weight = in_proj_weight_.weight()[{{0, embed_dim_}, kAll}];
    auto k_weight = in_proj_weight_.weight()[{{embed_dim_, 2 * embed_dim_}, kAll}];
    auto v_weight = in_proj_weight_.weight()[{{2 * embed_dim_, 3 * embed_dim_}, kAll}];

    auto q_bias = in_proj_bias_.weight()[{{0, embed_dim_}}];
    auto k_bias = in_proj_bias_.weight()[{{embed_dim_, 2 * embed_dim_}}];
    auto v_bias = in_proj_bias_.weight()[{{2 * embed_dim_, 3 * embed_dim_}}];

    auto q = nn::functional::matmul(query, q_weight, false, true);
    auto k = nn::functional::matmul(key, k_weight, false, true);
    auto v = nn::functional::matmul(value, v_weight, false, true);

    q = q + q_bias;
    k = k + k_bias;
    v = v + v_bias;

    auto q_reshaped = Tensor::empty({num_heads_, num_queries, head_dim_}, kFloat32).alloc();
    for (int nq = 0; nq < num_queries; nq++) {
      for (int h = 0; h < num_heads_; h++) {
        for (int d = 0; d < head_dim_; d++) {
          float val = q.at<float>({nq, h * head_dim_ + d});
          *q_reshaped.offsettedPtr<float>({h, nq, d}) = val;
        }
      }
    }
    q = q_reshaped;  // [num_heads, num_queries, head_dim]
    auto k_reshaped = Tensor::empty({num_heads_, seq_len, head_dim_}, kFloat32).alloc();
    for (int s = 0; s < seq_len; s++) {
      for (int h = 0; h < num_heads_; h++) {
        for (int d = 0; d < head_dim_; d++) {
          float val = k.at<float>({s, h * head_dim_ + d});
          *k_reshaped.offsettedPtr<float>({h, s, d}) = val;
        }
      }
    }
    k = k_reshaped;
    auto v_reshaped = Tensor::empty({num_heads_, seq_len, head_dim_}, kFloat32).alloc();
    for (int s = 0; s < seq_len; s++) {
      for (int h = 0; h < num_heads_; h++) {
        for (int d = 0; d < head_dim_; d++) {
          float val = v.at<float>({s, h * head_dim_ + d});
          *v_reshaped.offsettedPtr<float>({h, s, d}) = val;
        }
      }
    }
    v = v_reshaped;

    // q = q.view({num_queries, num_heads_, head_dim_}).transpose(0, 1).contiguous();  // [num_heads, num_queries, head_dim]
    // k = k.view({seq_len, num_heads_, head_dim_}).transpose(0, 1).contiguous();      // [num_heads, seq_len, head_dim]
    // v = v.view({seq_len, num_heads_, head_dim_}).transpose(0, 1).contiguous();      // [num_heads, seq_len, head_dim]

    auto scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto attn_weights = nn::functional::matmul(q, k, false, true) * scale;  // [num_heads, num_queries, seq_len]

    if (has_key_padding_mask && key_padding_mask.numel() > 0) {
      auto mask_value = -std::numeric_limits<float>::infinity();
      for (int32_t h = 0; h < num_heads_; ++h) {
        for (int32_t q_idx = 0; q_idx < num_queries; ++q_idx) {
          for (int32_t s = 0; s < seq_len; ++s) {
            if (key_padding_mask.at<uint8_t>({s}) == 1) { *attn_weights.offsettedPtr<float>({h, q_idx, s}) = mask_value; }
          }
        }
      }
    }

    attn_weights = nn::functional::softmax(attn_weights.unsqueeze(0), -1).squeeze(0);

    auto attn_output = nn::functional::matmul(attn_weights, v);  // [num_heads, num_queries, head_dim]

    auto attn_output_reshaped = Tensor::empty({num_queries, embed_dim_}, kFloat32).alloc();
    for (int h = 0; h < num_heads_; h++) {
      for (int nq = 0; nq < num_queries; nq++) {
        for (int d = 0; d < head_dim_; d++) {
          float val = attn_output.at<float>({h, nq, d});
          *attn_output_reshaped.offsettedPtr<float>({nq, h * head_dim_ + d}) = val;
        }
      }
    }
    attn_output = attn_output_reshaped;

    return {out_proj_(attn_output)};
  }
};

class Resampler : public nn::Module {
  int32_t num_queries_;
  int32_t embed_dim_;
  int32_t num_heads_;
  int32_t kv_dim_;
  std::vector<int32_t> max_size_;

  nn::Param query_;
  nn::Linear kv_proj_;
  ResamplerAttention attn_;
  nn::LayerNorm ln_q_;
  nn::LayerNorm ln_kv_;
  nn::LayerNorm ln_post_;
  nn::Param proj_;

 public:
  Resampler() = default;

  Resampler(const std::string& name, int32_t num_queries, int32_t embed_dim, int32_t num_heads, int32_t kv_dim = -1,
            const std::vector<int32_t>& max_size = {70, 70})
      : nn::Module(name),
        num_queries_(num_queries),
        embed_dim_(embed_dim),
        num_heads_(num_heads),
        kv_dim_(kv_dim == -1 ? embed_dim : kv_dim),
        max_size_(max_size) {
    query_ = reg<nn::Param>("query", getModuleName() + ".query", Tensor::shape_t{num_queries_, embed_dim_});
    proj_ = reg<nn::Param>("proj", getModuleName() + ".proj", Tensor::shape_t{embed_dim_, embed_dim_});

    // kv_proj: project from kv_dim (1152) to embed_dim (3584)
    kv_proj_ = reg<nn::Linear>("kv_proj", kv_dim_, embed_dim_, false);  // no bias

    attn_ = reg<ResamplerAttention>("attn", embed_dim_, num_heads_);
    ln_q_ = reg<nn::LayerNorm>("ln_q", std::vector<int32_t>{embed_dim_}, true, true, 1e-6);
    ln_kv_ = reg<nn::LayerNorm>("ln_kv", std::vector<int32_t>{embed_dim_}, true, true, 1e-6);
    ln_post_ = reg<nn::LayerNorm>("ln_post", std::vector<int32_t>{embed_dim_}, true, true, 1e-6);
    auto pos_embed = get2DSinCosPosEmbed(embed_dim_, max_size_);
    registerBuffer("pos_embed", pos_embed);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];          // [batch_size, seq_len, kv_dim] or [seq_len, kv_dim]
    auto tgt_sizes = inputs[1];  // Tensor with shape [batch_size, 2] each item is [h, w]

    auto batch_size = 1;
    auto seq_len = x.shape()[0];

    if (x.shape().size() == 3) {
      batch_size = x.shape()[0];
      seq_len = x.shape()[1];
    } else {
      x = x.unsqueeze(0);
    }

    std::vector<int> patch_len(batch_size);
    int max_h = 0, max_w = 0, max_patch_len = 0;
    for (int i = 0; i < batch_size; i++) {
      patch_len[i] = tgt_sizes.at<int>({i, 0}) * tgt_sizes.at<int>({i, 1});
      if (patch_len[i] > max_patch_len) max_patch_len = patch_len[i];
      if (tgt_sizes.at<int>({i, 0}) > max_h) max_h = tgt_sizes.at<int>({i, 0});
      if (tgt_sizes.at<int>({i, 1}) > max_w) max_w = tgt_sizes.at<int>({i, 1});
    }

    if (max_h > max_size_[0] || max_w > max_size_[1]) {
      max_size_[0] = max_h;
      max_size_[1] = max_w;
      auto new_pos_embed = get2DSinCosPosEmbed(embed_dim_, max_size_);
      registerBuffer("pos_embed", new_pos_embed);
    }

    auto pos_embed = getBuffer("pos_embed");  // [max_h, max_w, embed_dim]

    auto key_padding_mask = Tensor::empty({batch_size, max_patch_len}, kUInt8).alloc();
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < max_patch_len; j++) { key_padding_mask.at<uint8_t>({i, j}) = 1; }
      for (int j = 0; j < patch_len[i] && j < max_patch_len; j++) { key_padding_mask.at<uint8_t>({i, j}) = 0; }
    }

    std::vector<Tensor> pos_embed_list;

    for (int i = 0; i < batch_size; i++) {
      int32_t tgt_h = tgt_sizes.at<int32_t>({i, 0});
      int32_t tgt_w = tgt_sizes.at<int32_t>({i, 1});
      int32_t patch_count = tgt_h * tgt_w;

      Tensor pos_embed_i = Tensor::empty({patch_count, embed_dim_}, kFloat32).alloc();

      int patch_idx = 0;
      for (int h = 0; h < tgt_h; h++) {
        for (int w = 0; w < tgt_w; w++) {
          for (int d = 0; d < embed_dim_; d++) {
            float value = pos_embed.at<float>({h, w, d});
            *pos_embed_i.offsettedPtr<float>({patch_idx, d}) = value;
          }
          patch_idx++;
        }
      }

      pos_embed_list.push_back(pos_embed_i);
    }

    Tensor pos_embed_padded = Tensor::empty({batch_size, max_patch_len, embed_dim_}, kFloat32).alloc();
    for (int i = 0; i < batch_size; i++) {
      auto& pos_embed_i = pos_embed_list[i];
      int actual_len = pos_embed_i.shape()[0];

      for (int j = 0; j < actual_len && j < max_patch_len; j++) {
        for (int k = 0; k < embed_dim_; k++) {
          *pos_embed_padded.offsettedPtr<float>({i, j, k}) = pos_embed_i.at<float>({j, k});
        }
      }

      for (int j = actual_len; j < max_patch_len; j++) {
        for (int k = 0; k < embed_dim_; k++) { *pos_embed_padded.offsettedPtr<float>({i, j, k}) = 0.0f; }
      }
    }

    x = kv_proj_(x);

    x = ln_kv_(x);

    auto q = ln_q_(query_.weight());  // [num_queries, embed_dim]

    std::vector<Tensor> outputs;
    for (int32_t b = 0; b < batch_size; ++b) {
      // x for this batch
      Tensor x_b = x[make_slice(b), kAll, kAll].view({seq_len, embed_dim_});

      // pos_embed for this batch
      // Tensor pos_embed_b = Tensor::empty({seq_len, embed_dim_}, kFloat32).alloc();
      // for (int i = 0; i < seq_len; i++) {
      //   for (int j = 0; j < embed_dim_; j++) {
      //     if (i < max_patch_len) {
      //       pos_embed_b.at<float>({i, j}) = pos_embed_padded.at<float>({b, i, j});
      //     } else {
      //       pos_embed_b.at<float>({i, j}) = 0.0f;
      //     }
      //   }
      // }
      // TODO: handle 'set 0'
      Tensor pos_embed_b = pos_embed_padded[make_slice(b), kAll, kAll].view({seq_len, embed_dim_});

      auto kv_input = x_b + pos_embed_b;

      // key_padding_mask for this batch
      Tensor key_padding_mask_b = key_padding_mask[make_slice(b), kAll].view({max_patch_len});

      bool has_padding = false;
      for (int i = 0; i < seq_len; i++) {
        if (key_padding_mask_b.at<uint8_t>({i}) == 1) {
          has_padding = true;
          break;
        }
      }

      auto attn_output = has_padding ? attn_(q, kv_input, x_b, key_padding_mask_b)[0] : attn_(q, kv_input, x_b)[0];

      outputs.push_back(attn_output);
    }

    auto out_tensor = Tensor::empty({batch_size, num_queries_, embed_dim_}, kFloat32).alloc();
    // Optimize: Use memcpy for contiguous memory copy instead of nested loops
    const int32_t query_embed_size = num_queries_ * embed_dim_;
    for (int32_t i = 0; i < batch_size; i++) {
      auto& out_i = outputs[i];
      float* dst_ptr = out_tensor.offsettedPtr<float>({i, 0, 0});
      const float* src_ptr = out_i.ptr<float>();
      std::memcpy(dst_ptr, src_ptr, query_embed_size * sizeof(float));
    }

    out_tensor = ln_post_(out_tensor);

    auto original_shape = out_tensor.shape();
    auto reshaped = out_tensor.view({original_shape[0] * original_shape[1], original_shape[2]});
    reshaped = nn::functional::matmul(reshaped, proj_.weight());
    out_tensor = reshaped.view(original_shape);

    if (inputs[0].shape().size() == 2) { out_tensor = out_tensor.squeeze(0); }

    return {out_tensor};  //[batch_size, num_queries, embed_dim] or [num_queries, embed_dim]
  }
};

}  // namespace mllm::models::minicpmo
