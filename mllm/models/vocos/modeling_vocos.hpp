// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/ParameterFile.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/utils/Common.hpp"
#include <complex>
#include <cstdint>
#include <string>

namespace mllm::models::vocos {

/**
 * @brief ISTFTHead module for Vocos
 *
 * This module transforms features to STFT coefficients and applies inverse STFT
 */
class ISTFTHead : public nn::Module {
  int32_t n_fft_;
  int32_t hop_length_;

  nn::Linear out_;
  nn::ISTFT istft_;
  nn::Param window_;

 public:
  ISTFTHead() = default;

  /**
   * @brief Construct an ISTFTHead
   *
   * @param name Module name
   * @param dim Input dimension
   * @param n_fft FFT size
   * @param hop_length Hop length for STFT
   */
  inline ISTFTHead(const std::string& name, int32_t dim, int32_t n_fft, int32_t hop_length)
      : nn::Module(name), n_fft_(n_fft), hop_length_(hop_length) {
    out_ = reg<nn::Linear>("out", dim, n_fft + 2, true, aops::LinearImplTypes::kDefault);
    istft_ = reg<nn::ISTFT>("istft", n_fft_, hop_length_, n_fft_, true);
    window_ = reg<nn::Param>("istft.window", getModuleName() + ".istft.window", Tensor::shape_t{1, n_fft_});
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];  // [B, T, C]

    // Transform to STFT coefficients
    x = out_(x);            // [B, T, n_fft + 2]
    x = x.transpose(1, 2);  // [B, n_fft + 2, T]

    auto [mag, p] = nn::functional::chunk<2>(x, 1);

    mag = nn::functional::exp(mag);

    mag = mag.clip(std::numeric_limits<float>::min(), 1e2);

    x = nn::functional::cos(p);
    auto y = nn::functional::sin(p);

    auto S = mag * (x + std::complex<float>{0, 1} * y);

    auto audio = istft_(S, window_.weight());

    return {audio};
  }
};

/**
 * @brief ConvNeXtBlock module for Vocos backbone
 *
 * This block consists of:
 * 1. Depthwise convolution
 * 2. Layer normalization
 * 3. Pointwise convolutions with GELU activation
 */
class ConvNeXtBlock : public nn::Module {
  int32_t dim_;
  int32_t intermediate_dim_;

  nn::Conv1D dwconv_;   // Depthwise convolution (using Conv3D with 1D kernel)
  nn::LayerNorm norm_;  // Layer normalization
  nn::Linear pwconv1_;  // Pointwise convolution 1
  nn::GELU act_;        // GELU activation
  nn::Linear pwconv2_;  // Pointwise convolution 2
  nn::Param gamma_;     // Learnable parameter for scaling

 public:
  ConvNeXtBlock() = delete;

  /**
   * @brief Construct a ConvNeXtBlock
   *
   * @param name Module name
   * @param dim Input and output dimension
   * @param intermediate_dim Hidden dimension for pointwise convolutions
   */
  inline ConvNeXtBlock(const std::string& name, int32_t dim, int32_t intermediate_dim)
      : nn::Module(name), dim_(dim), intermediate_dim_(intermediate_dim) {
    dwconv_ = reg<nn::Conv1D>("dwconv", dim, dim, 7,
                              /*stride*/ 1, /*padding*/ 3, /*dilation*/ 1, /*groups*/ dim);  // Depthwise conv
    norm_ = reg<nn::LayerNorm>("norm", std::vector<int32_t>{dim}, true, true, 1e-6);
    pwconv1_ = reg<nn::Linear>("pwconv1", dim, intermediate_dim, true, aops::LinearImplTypes::kDefault);
    act_ = reg<nn::GELU>("act");
    pwconv2_ = reg<nn::Linear>("pwconv2", intermediate_dim, dim, true, aops::LinearImplTypes::kDefault);
    gamma_ = reg<nn::Param>("gamma", getModuleName() + ".gamma", std::vector<int32_t>{dim});
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];  // [B, C, T]

    auto residual = x;

    // Depthwise convolution

    x = dwconv_(x);

    x = x.transpose(1, 2);  // [B, T, C]
    x = norm_(x);
    x = pwconv1_(x);
    x = act_(x);
    x = pwconv2_(x);

    x = x * gamma_.weight();

    x = x.transpose(1, 2);  // [B, C, T]

    x = x + residual;

    return {x};
  }
};

/**
 * @brief VocosBackbone module
 *
 * This is the main backbone of the Vocos model based on ConvNeXt blocks
 */
class VocosBackbone final : public nn::Module {
  int32_t dim_;

  nn::Conv1D embed_;
  nn::LayerNorm norm_;
  nn::ModuleList<ConvNeXtBlock> convnext_;
  nn::LayerNorm final_layer_norm_;

 public:
  VocosBackbone() = default;

  /**
   * @brief Construct a VocosBackbone
   *
   * @param name Module name
   * @param dim Hidden dimension
   * @param intermediate_dim Intermediate dimension for ConvNeXt blocks
   * @param num_layers Number of ConvNeXt blocks
   */
  inline VocosBackbone(const std::string& name, int32_t dim, int32_t intermediate_dim, int32_t num_layers)
      : nn::Module(name), dim_(dim) {
    // Embedding convolution
    // Using Conv1D with kernel size [1, 1, kernel_size] to simulate 1D convolution
    embed_ = reg<nn::Conv1D>("embed", 100, dim, 7, 1, /*padding*/ 3);
    norm_ = reg<nn::LayerNorm>("norm", std::vector<int32_t>{dim}, true, true, 1e-6);

    // ConvNeXt blocks
    convnext_ = reg<nn::ModuleList<ConvNeXtBlock>>("convnext", num_layers, dim, intermediate_dim);

    // Final layer norm
    final_layer_norm_ = reg<nn::LayerNorm>("final_layer_norm", std::vector<int32_t>{dim}, true, true, 1e-6);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];  // [B, C, T]

    x = embed_(x);

    x = x.transpose(1, 2);  // [B, T, C]
    x = norm_(x);
    x = x.transpose(1, 2);  // [B, C, T]

    for (auto& block : convnext_.list()) { x = block(x)[0]; }

    x = final_layer_norm_(x.transpose(1, 2));

    return {x};
  }
};

/**
 * @brief Feature extractor for Vocos
 *
 * Converts raw audio to mel spectrogram features
 * Following but not a complete torch.nn.Spectrogram(no window normalization)
 */
class MelSpectrogramFeatures final : public nn::Module {
  int32_t n_fft_;
  int32_t hop_length_;
  int32_t win_length_;
  int32_t n_mels_;
  std::string padding_;
  int power_;
  nn::STFT stft_;

  nn::Param window_;
  nn::Param melscale_fbanks_;

 public:
  MelSpectrogramFeatures() = default;

  /**
   * @brief Construct MelSpectrogramFeatures
   * @param name Module name
   * @param sample_rate Sample rate of audio
   * @param n_fft Size of FFT
   * @param hop_length Distance between neighboring sliding window frames
   * @param n_mels Number of mel filter banks
   * @param padding Padding type, "center" or "same"
   * @param power Power of the mel scale, -1 for no scaling
   */
  explicit inline MelSpectrogramFeatures(const std::string& name, int32_t sample_rate = 24000, int32_t n_fft = 1024,
                                         int32_t hop_length = 256, int32_t n_mels = 100, const std::string& padding = "center",
                                         int power = 1)
      : nn::Module(name), n_fft_(n_fft), hop_length_(hop_length), n_mels_(n_mels), padding_(padding), power_(power) {
    if (padding != "center" && padding != "same") { throw std::invalid_argument("Padding must be 'center' or 'same'."); }

    win_length_ = n_fft;  // win_length defaults to n_fft

    // STFT layer
    stft_ = reg<nn::STFT>("stft", n_fft, hop_length, win_length_, true, true, "reflect", true);
    // STFT window weight in torchaudio.transforms.Spectrogram
    window_ = reg<nn::Param>("spectrogram.window", getModuleName() + ".spectrogram.window", Tensor::shape_t{1, win_length_});

    // Mel filter bank weights in torchaudio.transforms.MelScale
    melscale_fbanks_ =
        reg<nn::Param>("mel_scale.fb", getModuleName() + ".mel_scale.fb", Tensor::shape_t{n_fft / 2 + 1, n_mels_});
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto audio = inputs[0];  // [B, T]

    // Handle padding
    if (padding_ == "same") {
      // int32_t pad = win_length_ - hop_length_;
      // audio = pad(audio, (pad / 2, pad / 2), "reflect");
      NYI("apply same padding in MelSpectrogramFeatures not implemented");
    }

    auto stft_result = stft_(audio, window_.weight());  // [B, freq_bins, n_frames]

    if (power_ != 1) { NYI("power != 1(power == 2) not implemented, which should do spec_f.abs().pow(power)"); }

    auto specgram = stft_result.abs();

    auto mel_specgram = nn::functional::matmul(specgram.T(), melscale_fbanks_.weight()).T();  // [B, n_mels, n_frames]

    mel_specgram = nn::functional::log(nn::functional::clip(mel_specgram, 1e-7, std::numeric_limits<float>::max()));

    return {mel_specgram};
  }
};

/**
 * @brief Main Vocos model
 *
 * Consists of:
 * 1. Feature extractor (MelSpectrogramFeatures)
 * 2. Backbone (VocosBackbone)
 * 3. Head (ISTFTHead)
 */
class Vocos {
  mllm::ParameterFile::ptr_t param;
  MelSpectrogramFeatures feature_extractor_;
  VocosBackbone backbone_;
  ISTFTHead head_;

 public:
  Vocos() = default;

  /**
   * @brief Construct a Vocos model
   *
   * @param name Module name
   * @param dim Hidden dimension
   * @param intermediate_dim Intermediate dimension for ConvNeXt blocks
   * @param num_layers Number of ConvNeXt blocks
   * @param n_fft FFT size
   * @param hop_length Hop length for STFT
   * @param n_mels Number of mel filter banks
   * @param padding Padding type for STFT (only support 'same')
   */
  inline Vocos(const std::string& name, int32_t dim, int32_t intermediate_dim, int32_t num_layers, int32_t n_fft,
               int32_t hop_length, int32_t n_mels, const std::string& padding) {
    // Feature extractor - placeholder implementation
    feature_extractor_ = MelSpectrogramFeatures("feature_extractor.mel_spec", 24000, n_fft, hop_length, n_mels, padding);

    // Backbone
    backbone_ = VocosBackbone("backbone", dim, intermediate_dim, num_layers);

    // Head
    head_ = ISTFTHead("head", dim, n_fft, hop_length);
  }

  std::vector<Tensor> operator()(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
    auto x = inputs[0];  // [B, T] - raw audio or [B, C, T] - features

    // Feature extraction
    x = feature_extractor_(x)[0];  // [B, C, T]

    // Backbone processing
    x = backbone_(x)[0];  // [B, C, T]

    // Head for waveform generation
    x = head_(x)[0];  // [B, n_fft + 2, T]

    return {x};
  }

  std::vector<Tensor> decode(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
    auto x = inputs[0];   // [B, C, T] - features
    x = backbone_(x)[0];  // [B, C, T]
    // Head for waveform generation
    x = head_(x)[0];  // [B, n_fft + 2, T]
    return {x};
  }

  void from_pretrained(const std::string& path, mllm::ModelFileVersion version = ModelFileVersion::kV1) {
    param = mllm::load(path, version);

    feature_extractor_.load(param);
    backbone_.load(param);
    head_.load(param);
  }
};

}  // namespace mllm::models::vocos