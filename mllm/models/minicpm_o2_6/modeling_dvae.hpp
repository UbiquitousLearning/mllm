// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/models/minicpm_o2_6/modeling_vector_quantize.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/utils/Common.hpp"
#include <cstdint>
#include <string>

namespace mllm::models::dvae {

/**
 * @brief ConvNeXtBlock module for DVAE
 *
 * This block consists of:
 * 1. Depthwise convolution
 * 2. Layer normalization
 * 3. Pointwise convolutions with GELU activation
 */
class ConvNeXtBlock : public nn::Module {
  int32_t dim_;
  int32_t intermediate_dim_;
  int32_t kernel_;
  int32_t dilation_;

  nn::Conv1D dwconv_;   // Depthwise convolution
  nn::LayerNorm norm_;  // Layer normalization
  nn::Linear pwconv1_;  // Pointwise convolution 1
  nn::GELU act_;        // GELU activation
  nn::Linear pwconv2_;  // Pointwise convolution 2
  nn::Param coef_;      // Learnable parameter for scaling

 public:
  ConvNeXtBlock() = delete;

  /**
   * @brief Construct a ConvNeXtBlock
   *
   * @param name Module name
   * @param dim Input and output dimension
   * @param intermediate_dim Hidden dimension for pointwise convolutions
   * @param kernel Kernel size for depthwise convolution
   * @param dilation Dilation for depthwise convolution
   * @param layer_scale_init_value Initial value for layer scaling coefficient
   */
  inline ConvNeXtBlock(const std::string& name, int32_t dim, int32_t intermediate_dim, int32_t kernel, int32_t dilation,
                       float layer_scale_init_value = 1e-6)
      : nn::Module(name), dim_(dim), intermediate_dim_(intermediate_dim), kernel_(kernel), dilation_(dilation) {
    // Depthwise conv with padding = dilation * (kernel // 2)
    int32_t padding = dilation * (kernel / 2);
    dwconv_ = reg<nn::Conv1D>("dwconv", dim, dim, kernel, /*stride*/ 1, padding, dilation, dim);  // Depthwise conv
    norm_ = reg<nn::LayerNorm>("norm", std::vector<int32_t>{dim}, true, true, 1e-6);
    pwconv1_ = reg<nn::Linear>("pwconv1", dim, intermediate_dim, true, aops::LinearImplTypes::kDefault);
    act_ = reg<nn::GELU>("act");
    pwconv2_ = reg<nn::Linear>("pwconv2", intermediate_dim, dim, true, aops::LinearImplTypes::kDefault);

    if (layer_scale_init_value > 0) { coef_ = reg<nn::Param>("coef", getModuleName() + ".coef", std::vector<int32_t>{dim}); }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];  // [B, C, T]
    auto residual = x;

    // Depthwise convolution
    x = dwconv_(x);
    x = x.transpose(1, 2);  // [B, C, T] -> [B, T, C]

    x = norm_(x);
    x = pwconv1_(x);
    x = act_(x);
    x = pwconv2_(x);

    if (!coef_.weight().isNil()) { x = x * coef_.weight(); }

    x = x.transpose(1, 2);  // [B, T, C] -> [B, C, T]
    x = x + residual;

    return {x};
  }
};

/**
 * @brief GFSQ (Grouped Feature Quantization) module for DVAE
 */
class GFSQ : public nn::Module {
  int32_t dim_;
  std::vector<int32_t> levels_;
  int32_t G_;  // Groups
  int32_t R_;  // Quantizers
  float eps_;
  bool transpose_;

  // Note: GroupedResidualFSQ implementation would be needed here
  // For now, we'll leave it as a placeholder since it's a complex component
  vq::GroupedResidualFSQ quantizer_;  // Placeholder for GroupedResidualFSQ
  int32_t n_ind_;

 public:
  GFSQ() = default;

  /**
   * @brief Construct a GFSQ module
   *
   * @param name Module name
   * @param dim Dimension of the input features
   * @param levels Levels for quantization
   * @param G Number of groups
   * @param R Number of quantizers
   * @param eps Epsilon value for numerical stability
   * @param transpose Whether to transpose input
   */
  inline GFSQ(const std::string& name, int32_t dim, const std::vector<int32_t>& levels, int32_t G, int32_t R, float eps = 1e-5,
              bool transpose = true)
      : nn::Module(name), dim_(dim), levels_(levels), G_(G), R_(R), eps_(eps), transpose_(transpose) {
    // Calculate n_ind = math.prod(levels)
    n_ind_ = 1;
    for (int level : levels) { n_ind_ *= level; }

    quantizer_ = reg<vq::GroupedResidualFSQ>("quantizer", dim_, levels, R, G_);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    NYI("not verified");
    auto x = inputs[0];  // [B, C, T]
    if (transpose_) {
      x = x.transpose(1, 2);  // [B, C, T] -> [B, T, C]
    }
    auto ind = quantizer_(x)[1];
    ind = ind.permute({1, 2, 0, 3}).contiguous();
    ind = ind.view({ind.shape()[0], ind.shape()[1], -1});  // [B, C, T] -> [B, C, R*G]
    if (transpose_) {
      ind = ind.transpose(1, 2);  // [B, T, C] -> [B, C, T]
    }
    return {ind};
  }

  /**
   * @brief Embed indices to feature vectors
   *
   * @param x Input indices tensor
   * @return Embedded features
   */
  Tensor embed(const Tensor& x) {
    auto y = x;
    if (transpose_) {
      y = y.transpose(1, 2);  // [B, C, T] -> [B, T, C]
    }
    // Reshape and permute to match the expected format
    y = y.view({y.shape()[0], y.shape()[1], G_, R_}).permute({2, 0, 1, 3});

    auto feat = quantizer_.getOutputFromIndices(y);

    if (transpose_) {
      feat = feat.transpose(1, 2);  // [B, T, C] -> [B, C, T]
    }
    return feat;
  }
};

/**
 * @brief DVAE Decoder module
 */
class DVAEDecoder : public nn::Module {
  bool up_;
  nn::ModuleList<ConvNeXtBlock> decoder_block_;
  nn::Conv1D conv_in_1_;
  nn::Conv1D conv_in_2_;
  nn::GELU gelu_;
  nn::Conv1D conv_out_;

 public:
  DVAEDecoder() = default;

  /**
   * @brief Construct a DVAEDecoder
   *
   * @param name Module name
   * @param idim Input dimension
   * @param odim Output dimension
   * @param n_layer Number of ConvNeXt blocks
   * @param bn_dim Bottleneck dimension
   * @param hidden Hidden dimension
   * @param kernel Kernel size
   * @param dilation Dilation factor
   * @param up Whether to upsample
   */
  inline DVAEDecoder(const std::string& name, int32_t idim, int32_t odim, int32_t n_layer = 12, int32_t bn_dim = 64,
                     int32_t hidden = 256, int32_t kernel = 7, int32_t dilation = 2, bool up = false)
      : nn::Module(name), up_(up) {
    // Input convolution layers
    conv_in_1_ = reg<nn::Conv1D>("conv_in.0", idim, bn_dim, 3, 1, 1);
    gelu_ = reg<nn::GELU>("conv_in.1");
    conv_in_2_ = reg<nn::Conv1D>("conv_in.2", bn_dim, hidden, 3, 1, 1);

    // Decoder blocks
    decoder_block_ = reg<nn::ModuleList<ConvNeXtBlock>>("decoder_block", n_layer, hidden, hidden * 4, kernel, dilation);

    // Output convolution
    conv_out_ =
        reg<nn::Conv1D>("conv_out", hidden, odim, 1, /*stride*/ 1, /*padding*/ 0, /*dilation*/ 1, /*groups*/ 1, /*bias*/ false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];  // [B, C, T]

    // Apply input convolutions
    x = conv_in_1_(x);
    x = gelu_(x);
    x = conv_in_2_(x);

    // Apply decoder blocks
    for (auto& block : decoder_block_.list()) { x = block(x)[0]; }

    // Apply output convolution
    x = conv_out_(x);

    return {x};
  }
};

/**
 * @brief DVAE (Discrete Variational Autoencoder) module
 */
class DVAE : public nn::Module {
  nn::Param coef_;
  nn::Conv1D downsample_conv_1_;
  nn::Conv1D downsample_conv_2_;
  nn::GELU gelu_1_;
  nn::GELU gelu_2_;

  // Unique pointers for optional components
  DVAEDecoder encoder_;
  DVAEDecoder decoder_;
  nn::Conv1D out_conv_;
  GFSQ vq_layer_;

 public:
  DVAE() = default;
  explicit DVAE(const std::string& name) : Module(name) {
    // Coefficient parameter
    coef_ = reg<nn::Param>("coef", getModuleName() + ".coef", Tensor::shape_t{1, 100, 1});

    // Downsample convolution layers
    downsample_conv_1_ = reg<nn::Conv1D>("downsample_conv.0", 100, 512, 3, 1, 1);
    gelu_1_ = reg<nn::GELU>("downsample_conv.1");
    downsample_conv_2_ = reg<nn::Conv1D>("downsample_conv.2", 512, 512, 4, 2, 1);
    gelu_2_ = reg<nn::GELU>("downsample_conv.3");

    // Encoder
    encoder_ = reg<DVAEDecoder>("encoder", 512, 1024, 12, 128, 256);

    // Decoder
    decoder_ = reg<DVAEDecoder>("decoder", 512, 512, 12, 128, 256);

    // Output convolution
    out_conv_ = reg<nn::Conv1D>("out_conv", 512, 100, 3, 1, /*padding*/ 1, /*dilation*/ 1, /*groups*/ 1, /*bias*/ false);

    // VQ layer
    vq_layer_ = reg<GFSQ>("vq_layer", 1024, std::vector<int32_t>{5, 5, 5, 5}, 2, 2);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto inp = inputs[0];  // [B, C, T]
    auto isEncode = (args.size() > 0 && args[0].get<bool>());

    if (isEncode && encoder_.impl() && vq_layer_.impl()) {
      auto mel = inp.clone();
      // Divide mel by coef
      mel = mel / coef_.weight();
      // Apply downsample convolution
      auto x = downsample_conv_1_(mel);
      x = gelu_1_(x);
      x = downsample_conv_2_(x);
      x = gelu_2_(x);
      // Add dummy dimension
      x = x.unsqueeze(0);
      // Apply encoder
      x = encoder_(x)[0];
      // Apply VQ layer
      auto ind = vq_layer_(x)[0];
      return {ind};
    }

    Tensor vq_feats;
    if (vq_layer_.impl()) {
      vq_feats = vq_layer_.embed(inp);
    } else {
      vq_feats = inp;
    }

    // Reshape vq_feats
    auto B = vq_feats.shape()[0];
    auto C = vq_feats.shape()[1];
    auto T = vq_feats.shape()[2];

    // perform: vq_feats.view({B, 2, C / 2, T}).permute({0, 2, 3, 1}).flatten(2);
    vq_feats = vq_feats.view({B, 2, C / 2, T});
    vq_feats = vq_feats.permute({0, 2, 3, 1});
    vq_feats = vq_feats.view({B, vq_feats.shape()[1], vq_feats.shape()[2] * vq_feats.shape()[3]});

    // Apply decoder
    auto dec_out = decoder_(vq_feats)[0];

    // Apply output convolution
    dec_out = out_conv_(dec_out);

    // Multiply by coef
    dec_out = dec_out * coef_.weight();

    return {dec_out};
  }
};

}  // namespace mllm::models::dvae