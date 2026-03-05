// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/preprocessor/audio/Audio.hpp"

namespace mllm::models::qwen2_5omni {

inline float hertz_to_mel_slaney(float freq) {
  constexpr float kMinLogHertz = 1000.0f;
  constexpr float kMinLogMel = 15.0f;
  const float logstep = 27.0f / std::log(6.4f);

  if (freq < kMinLogHertz) {
    return 3.0f * freq / 200.0f;
  }
  return kMinLogMel + std::log(freq / kMinLogHertz) * logstep;
}

inline float mel_to_hertz_slaney(float mel) {
  constexpr float kMinLogHertz = 1000.0f;
  constexpr float kMinLogMel = 15.0f;
  const float logstep = std::log(6.4f) / 27.0f;

  if (mel < kMinLogMel) {
    return 200.0f * mel / 3.0f;
  }
  return kMinLogHertz * std::exp(logstep * (mel - kMinLogMel));
}

inline Tensor create_hann_window(int32_t window_length, bool periodic = true) {
  int32_t length = periodic ? window_length + 1 : window_length;
  auto window = Tensor::empty({1, window_length}, kFloat32, kCPU).alloc();
  float* window_ptr = window.ptr<float>();

  for (int32_t i = 0; i < window_length; ++i) {
    float n = static_cast<float>(i);
    float denominator = periodic ? static_cast<float>(length) : static_cast<float>(length - 1);
    window_ptr[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * n / denominator);
  }

  return window;
}

inline Tensor create_mel_filterbank(int32_t num_frequency_bins, int32_t num_mel_filters, float min_frequency,
                                    float max_frequency, int32_t sampling_rate) {
  std::vector<float> fft_freqs(num_frequency_bins);
  for (int32_t i = 0; i < num_frequency_bins; ++i) {
    fft_freqs[i] = static_cast<float>(i) * (sampling_rate / 2.0f) / (num_frequency_bins - 1);
  }

  float mel_min = hertz_to_mel_slaney(min_frequency);
  float mel_max = hertz_to_mel_slaney(max_frequency);

  std::vector<float> mel_freqs(num_mel_filters + 2);
  for (int32_t i = 0; i < num_mel_filters + 2; ++i) {
    mel_freqs[i] = mel_min + static_cast<float>(i) * (mel_max - mel_min) / (num_mel_filters + 1);
  }

  std::vector<float> filter_freqs(num_mel_filters + 2);
  for (int32_t i = 0; i < num_mel_filters + 2; ++i) { filter_freqs[i] = mel_to_hertz_slaney(mel_freqs[i]); }

  auto mel_filters = Tensor::empty({num_frequency_bins, num_mel_filters}, kFloat32, kCPU).alloc();
  float* filters_ptr = mel_filters.ptr<float>();
  std::fill_n(filters_ptr, num_frequency_bins * num_mel_filters, 0.0f);

  for (int32_t mel_idx = 0; mel_idx < num_mel_filters; ++mel_idx) {
    float left_freq = filter_freqs[mel_idx];
    float center_freq = filter_freqs[mel_idx + 1];
    float right_freq = filter_freqs[mel_idx + 2];

    for (int32_t freq_idx = 0; freq_idx < num_frequency_bins; ++freq_idx) {
      float freq = fft_freqs[freq_idx];
      float value = 0.0f;

      if (freq >= left_freq && freq <= center_freq && center_freq != left_freq) {
        value = (freq - left_freq) / (center_freq - left_freq);
      } else if (freq >= center_freq && freq <= right_freq && right_freq != center_freq) {
        value = (right_freq - freq) / (right_freq - center_freq);
      }

      filters_ptr[freq_idx * num_mel_filters + mel_idx] = value;
    }
  }

  for (int32_t mel_idx = 0; mel_idx < num_mel_filters; ++mel_idx) {
    float enorm = 2.0f / (filter_freqs[mel_idx + 2] - filter_freqs[mel_idx]);
    for (int32_t freq_idx = 0; freq_idx < num_frequency_bins; ++freq_idx) {
      filters_ptr[freq_idx * num_mel_filters + mel_idx] *= enorm;
    }
  }

  return mel_filters;
}

class MelSpectrogramFeatures final : public nn::Module {
  int32_t n_fft_;
  int32_t hop_length_;
  int32_t win_length_;
  int32_t n_mels_;
  std::string padding_;
  int power_;
  nn::STFT stft_;
  Tensor window_;
  Tensor melscale_fbanks_;

 public:
  MelSpectrogramFeatures() = default;

  explicit inline MelSpectrogramFeatures(const std::string& name, int32_t sample_rate = 16000, int32_t n_fft = 400,
                                         int32_t hop_length = 160, int32_t n_mels = 128,
                                         const std::string& padding = "center", int power = 2)
      : nn::Module(name), n_fft_(n_fft), hop_length_(hop_length), n_mels_(n_mels), padding_(padding), power_(power) {
    if (padding != "center" && padding != "same") { throw std::invalid_argument("Padding must be 'center' or 'same'."); }

    win_length_ = n_fft_;
    stft_ = reg<nn::STFT>("stft", n_fft_, hop_length_, win_length_, true, true, "reflect", true);
    window_ = create_hann_window(win_length_, true);
    melscale_fbanks_ = create_mel_filterbank(n_fft_ / 2 + 1, n_mels_, 0.0f, 8000.0f, sample_rate);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto audio = inputs[0];  // [B, T]

    if (padding_ == "same") {
      NYI("apply same padding in MelSpectrogramFeatures not implemented");
    }

    auto stft_result = stft_(audio, window_);
    auto specgram = stft_result.abs();
    if (power_ == 2) {
      specgram = specgram * specgram;
    } else if (power_ != 1) {
      NYI("power != 1 and power != 2 not implemented");
    }

    auto mel_specgram = nn::functional::matmul(specgram.T(), melscale_fbanks_).T();
    mel_specgram = nn::functional::clip(mel_specgram, 1e-10f, std::numeric_limits<float>::max());
    mel_specgram = nn::functional::log(mel_specgram) / std::log(10.0f);
    auto max_val = mel_specgram.max();
    float threshold = max_val.item<float>() - 8.0f;
    mel_specgram = nn::functional::clip(mel_specgram, threshold, std::numeric_limits<float>::max());
    mel_specgram = (mel_specgram + 4.0f) / 4.0f;

    return {mel_specgram};
  }
};

struct Qwen2_5OmniAudioFeatures {
  Tensor input_features = Tensor::nil();
  int32_t feature_length = 0;
};

class Qwen2_5OmniAudioPreprocessor {
  MelSpectrogramFeatures mel_extractor_;
  int32_t sample_rate_;
  int32_t n_mels_;
  int32_t hop_length_;
  int32_t chunk_length_;
  int32_t n_samples_;

 public:
  explicit Qwen2_5OmniAudioPreprocessor(int32_t sample_rate = 16000, int32_t n_mels = 128, int32_t hop_length = 160,
                                        int32_t chunk_length = 300)
      : mel_extractor_("feature_extractor.mel_spec", sample_rate, 400, hop_length, n_mels, "center", 2),
        sample_rate_(sample_rate),
        n_mels_(n_mels),
        hop_length_(hop_length),
        chunk_length_(chunk_length),
        n_samples_(chunk_length * sample_rate) {}

  [[nodiscard]] Qwen2_5OmniAudioFeatures processAudioFile(const std::string& audio_file_path) {
    auto audio_data = mllm::audio::readWAV(audio_file_path, sample_rate_);
    if (audio_data.empty()) { return {}; }
    return processAudioData(audio_data.data(), static_cast<int32_t>(audio_data.size()));
  }

  [[nodiscard]] Qwen2_5OmniAudioFeatures processAudioData(const float* audio_data, int32_t audio_length) {
    Qwen2_5OmniAudioFeatures result;
    if (audio_data == nullptr || audio_length <= 0) { return result; }

    int32_t padded_length = n_samples_;
    int32_t effective_length = std::min(audio_length, padded_length);

    auto audio_tensor = Tensor::empty({1, padded_length}, kFloat32, kCPU).alloc();
    float* audio_ptr = audio_tensor.ptr<float>();

    if (audio_length <= padded_length) {
      std::memcpy(audio_ptr, audio_data, audio_length * sizeof(float));
      std::fill(audio_ptr + audio_length, audio_ptr + padded_length, 0.0f);
    } else {
      std::memcpy(audio_ptr, audio_data, padded_length * sizeof(float));
    }

    auto mel_spec = mel_extractor_.forward({audio_tensor}, {})[0];  // [1, n_mels, n_frames]

    int32_t valid_frames = calcFeatureLength(effective_length);
    int32_t max_frames = mel_spec.shape()[2];
    if (valid_frames > max_frames) { valid_frames = max_frames; }
    if (valid_frames <= 0) { return result; }

    auto trimmed = Tensor::empty({1, n_mels_, valid_frames}, kFloat32, kCPU).alloc();
    for (int32_t m = 0; m < n_mels_; ++m) {
      auto src_ptr = mel_spec.offsettedPtr<float>({0, m, 0});
      auto dst_ptr = trimmed.offsettedPtr<float>({0, m, 0});
      std::memcpy(dst_ptr, src_ptr, valid_frames * sizeof(float));
    }

    result.input_features = trimmed;
    result.feature_length = valid_frames;
    return result;
  }

  [[nodiscard]] int32_t calcFeatureLength(int32_t audio_length) const {
    if (audio_length <= 0) { return 0; }
    return (audio_length + hop_length_ - 1) / hop_length_;
  }

  [[nodiscard]] int32_t calcAudioTokenLength(int32_t feature_length) const {
    if (feature_length <= 0) { return 0; }
    int32_t after_conv = (feature_length - 1) / 2 + 1;
    if (after_conv < 2) { return 0; }
    int32_t after_pool = (after_conv - 2) / 2 + 1;
    return std::max(0, after_pool);
  }
};

}  // namespace mllm::models::qwen2_5omni
