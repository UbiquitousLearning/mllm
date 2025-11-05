// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"
#include <cmath>

namespace mllm::models::minicpmo {

// Whisper Encoder Attention
class WhisperEncoderAttention : public nn::Module {
  int32_t embed_dim_;
  int32_t num_heads_;
  int32_t head_dim_;
  float dropout_;

  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear q_proj_;
  nn::Linear out_proj_;

 public:
  WhisperEncoderAttention() = default;

  WhisperEncoderAttention(const std::string& name, int32_t embed_dim, int32_t num_heads, float dropout = 0.0f)
      : nn::Module(name), embed_dim_(embed_dim), num_heads_(num_heads), dropout_(dropout) {
    head_dim_ = embed_dim_ / num_heads_;

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
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto attn_weights = nn::functional::matmul(query_states, key_states.transpose(-2, -1)) * scale;

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

// Whisper Encoder Layer
class WhisperEncoderLayer : public nn::Module {
  int32_t embed_dim_;
  int32_t ffn_dim_;
  float dropout_;
  float activation_dropout_;

  WhisperEncoderAttention self_attn_;
  nn::LayerNorm self_attn_layer_norm_;
  nn::Linear fc1_;
  nn::Linear fc2_;
  nn::LayerNorm final_layer_norm_;
  nn::GELU activation_fn_;

 public:
  WhisperEncoderLayer() = default;

  WhisperEncoderLayer(const std::string& name, const MiniCPMOConfig& config) : nn::Module(name) {
    embed_dim_ = config.audio_hidden_size;
    // Whisper typically uses 4x hidden size for FFN
    ffn_dim_ = config.audio_hidden_size * 4;
    dropout_ = 0.0f;             // Default, can be added to config
    activation_dropout_ = 0.0f;  // Default

    self_attn_ = reg<WhisperEncoderAttention>("self_attn", embed_dim_, config.audio_num_attention_heads, dropout_);
    self_attn_layer_norm_ = reg<nn::LayerNorm>("self_attn_layer_norm", std::vector<int32_t>{embed_dim_}, true, true, 1e-5);
    fc1_ = reg<nn::Linear>("fc1", embed_dim_, ffn_dim_, true);
    fc2_ = reg<nn::Linear>("fc2", ffn_dim_, embed_dim_, true);
    final_layer_norm_ = reg<nn::LayerNorm>("final_layer_norm", std::vector<int32_t>{embed_dim_}, true, true, 1e-5);
    activation_fn_ = reg<nn::GELU>("activation_fn");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto attention_mask = inputs.size() > 1 ? inputs[1] : Tensor::nil();

    // Self attention with pre-norm
    auto residual = hidden_states;
    hidden_states = self_attn_layer_norm_(hidden_states);
    hidden_states = self_attn_(hidden_states, attention_mask)[0];
    // Dropout would be applied here in training
    hidden_states = residual + hidden_states;

    // FFN with pre-norm
    residual = hidden_states;
    hidden_states = final_layer_norm_(hidden_states);
    hidden_states = fc1_(hidden_states);
    hidden_states = activation_fn_(hidden_states);
    // Activation dropout would be applied here in training
    hidden_states = fc2_(hidden_states);
    // Dropout would be applied here in training
    hidden_states = residual + hidden_states;

    return {hidden_states};
  }
};

// Whisper Encoder
class WhisperEncoder : public nn::Module {
  int32_t embed_dim_;
  int32_t num_mel_bins_;
  int32_t max_source_positions_;
  int32_t num_layers_;

  nn::Conv1D conv1_;
  nn::GELU gelu1_;
  nn::Conv1D conv2_;
  nn::GELU gelu2_;
  nn::Embedding embed_positions_;
  std::vector<WhisperEncoderLayer> layers_;
  nn::LayerNorm layer_norm_;

 public:
  WhisperEncoder() = default;

  WhisperEncoder(const std::string& name, const MiniCPMOConfig& config) : nn::Module(name) {
    embed_dim_ = config.audio_hidden_size;
    num_mel_bins_ = 80;  // Standard for Whisper (can be added to config)
    max_source_positions_ = config.audio_max_position_embeddings;
    num_layers_ = config.audio_num_hidden_layers;

    // Convolutional input layers (Whisper uses Conv1D for feature extraction)
    // Conv1: (80, 1024, kernel_size=3, padding=1)
    conv1_ = reg<nn::Conv1D>("conv1", num_mel_bins_, embed_dim_, 3, 1, 1);
    gelu1_ = reg<nn::GELU>("gelu1");
    // Conv2: (1024, 1024, kernel_size=3, stride=2, padding=1)
    conv2_ = reg<nn::Conv1D>("conv2", embed_dim_, embed_dim_, 3, 2, 1);
    gelu2_ = reg<nn::GELU>("gelu2");

    // Position embedding
    embed_positions_ = reg<nn::Embedding>("embed_positions", max_source_positions_, embed_dim_);

    // Encoder layers
    for (int i = 0; i < num_layers_; ++i) {
      layers_.push_back(reg<WhisperEncoderLayer>("layers." + std::to_string(i), config));
    }

    // Final layer norm
    layer_norm_ = reg<nn::LayerNorm>("layer_norm", std::vector<int32_t>{embed_dim_}, true, true, 1e-5);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto input_features = inputs[0];  // [batch_size, num_mel_bins, sequence_length]
    auto attention_mask = inputs.size() > 1 ? inputs[1] : Tensor::nil();

    auto batch_size = input_features.shape()[0];

    // Apply Conv1D layers with GELU activation
    // Note: Conv1D expects [batch, in_channels, seq_len] format
    auto hidden_states = conv1_(input_features);
    hidden_states = gelu1_(hidden_states);
    hidden_states = conv2_(hidden_states);
    hidden_states = gelu2_(hidden_states);

    // Transpose to [batch, seq_len, embed_dim]
    hidden_states = hidden_states.transpose(1, 2);

    auto seq_len = hidden_states.shape()[1];

    // Add positional embeddings
    auto position_ids = Tensor::arange(0, seq_len, kInt64).view({1, seq_len});
    auto position_embeddings = embed_positions_(position_ids);
    hidden_states = hidden_states + position_embeddings;

    // Pass through encoder layers
    for (auto& layer : layers_) { hidden_states = layer(hidden_states, attention_mask)[0]; }

    // Final layer norm
    hidden_states = layer_norm_(hidden_states);

    return {hidden_states};
  }
};

}  // namespace mllm::models::minicpmo