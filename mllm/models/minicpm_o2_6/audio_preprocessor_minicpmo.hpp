// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/preprocessor/audio/Audio.hpp"

namespace mllm::models::minicpmo {

// Convert frequency from hertz to mel scale (Slaney implementation)
inline float hertz_to_mel_slaney(float freq) {
  constexpr float min_log_hertz = 1000.0f;
  constexpr float min_log_mel = 15.0f;
  const float logstep = 27.0f / std::log(6.4f);

  if (freq < min_log_hertz) {
    return 3.0f * freq / 200.0f;
  } else {
    return min_log_mel + std::log(freq / min_log_hertz) * logstep;
  }
}

// Convert frequency from mel scale to hertz (Slaney implementation)
inline float mel_to_hertz_slaney(float mel) {
  constexpr float min_log_hertz = 1000.0f;
  constexpr float min_log_mel = 15.0f;
  const float logstep = std::log(6.4f) / 27.0f;

  if (mel < min_log_mel) {
    return 200.0f * mel / 3.0f;
  } else {
    return min_log_hertz * std::exp(logstep * (mel - min_log_mel));
  }
}

// Create a Hann window
inline Tensor create_hann_window(int32_t window_length, bool periodic = true) {
  int32_t length = periodic ? window_length + 1 : window_length;
  auto window = Tensor::empty({1, window_length}, kFloat32, kCPU).alloc();
  float* window_ptr = window.ptr<float>();

  for (int32_t i = 0; i < window_length; ++i) {
    // Hann window formula: 0.5 - 0.5 * cos(2π * n / (N-1))
    // For periodic: use N instead of N-1
    float n = static_cast<float>(i);
    float denominator = periodic ? static_cast<float>(length) : static_cast<float>(length - 1);
    window_ptr[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * n / denominator);
  }

  return window;
}

// Create mel filter bank matrix
inline Tensor create_mel_filterbank(int32_t num_frequency_bins, int32_t num_mel_filters, float min_frequency,
                                    float max_frequency, int32_t sampling_rate) {
  // Create FFT frequencies
  std::vector<float> fft_freqs(num_frequency_bins);
  for (int32_t i = 0; i < num_frequency_bins; ++i) {
    fft_freqs[i] = static_cast<float>(i) * (sampling_rate / 2.0f) / (num_frequency_bins - 1);
  }

  // Convert min/max frequencies to mel scale
  float mel_min = hertz_to_mel_slaney(min_frequency);
  float mel_max = hertz_to_mel_slaney(max_frequency);

  // Create mel-spaced filter center frequencies
  std::vector<float> mel_freqs(num_mel_filters + 2);
  for (int32_t i = 0; i < num_mel_filters + 2; ++i) {
    mel_freqs[i] = mel_min + static_cast<float>(i) * (mel_max - mel_min) / (num_mel_filters + 1);
  }

  // Convert mel frequencies back to Hz
  std::vector<float> filter_freqs(num_mel_filters + 2);
  for (int32_t i = 0; i < num_mel_filters + 2; ++i) { filter_freqs[i] = mel_to_hertz_slaney(mel_freqs[i]); }

  // Create triangular filter bank matrix
  auto mel_filters = Tensor::empty({num_frequency_bins, num_mel_filters}, kFloat32, kCPU).alloc();
  float* filters_ptr = mel_filters.ptr<float>();

  // Initialize to zeros
  std::fill_n(filters_ptr, num_frequency_bins * num_mel_filters, 0.0f);

  // Fill in triangular filters
  for (int32_t mel_idx = 0; mel_idx < num_mel_filters; ++mel_idx) {
    float left_freq = filter_freqs[mel_idx];
    float center_freq = filter_freqs[mel_idx + 1];
    float right_freq = filter_freqs[mel_idx + 2];

    for (int32_t freq_idx = 0; freq_idx < num_frequency_bins; ++freq_idx) {
      float freq = fft_freqs[freq_idx];
      float value = 0.0f;

      // Left slope (rising)
      if (freq >= left_freq && freq <= center_freq && center_freq != left_freq) {
        value = (freq - left_freq) / (center_freq - left_freq);
      }
      // Right slope (falling)
      else if (freq >= center_freq && freq <= right_freq && right_freq != center_freq) {
        value = (right_freq - freq) / (right_freq - center_freq);
      }

      filters_ptr[freq_idx * num_mel_filters + mel_idx] = value;
    }
  }

  // Apply Slaney-style normalization (constant energy per channel)
  for (int32_t mel_idx = 0; mel_idx < num_mel_filters; ++mel_idx) {
    float enorm = 2.0f / (filter_freqs[mel_idx + 2] - filter_freqs[mel_idx]);
    for (int32_t freq_idx = 0; freq_idx < num_frequency_bins; ++freq_idx) {
      filters_ptr[freq_idx * num_mel_filters + mel_idx] *= enorm;
    }
  }

  return mel_filters;
}

/**
 * Whisper-style Mel-Spectrogram Feature Extractor
 * Converts audio to log mel-spectrogram features for Whisper encoder
 */
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

  explicit inline MelSpectrogramFeatures(const std::string& name, int32_t sample_rate = 24000, int32_t n_fft = 1024,
                                         int32_t hop_length = 256, int32_t n_mels = 100, const std::string& padding = "center",
                                         int power = 1)
      : nn::Module(name), n_fft_(n_fft), hop_length_(hop_length), n_mels_(n_mels), padding_(padding), power_(power) {
    if (padding != "center" && padding != "same") { throw std::invalid_argument("Padding must be 'center' or 'same'."); }

    win_length_ = n_fft;  // win_length defaults to n_fft

    // STFT layer
    stft_ = reg<nn::STFT>("stft", n_fft, hop_length, win_length_, true, true, "reflect", true);

    // Compute Hann window (matches np.hanning behavior)
    window_ = create_hann_window(win_length_, true);

    // Compute mel filterbank using Slaney scale (matches Whisper's configuration)
    // Whisper uses: min_freq=0.0, max_freq=8000.0, norm="slaney", mel_scale="slaney"
    melscale_fbanks_ = create_mel_filterbank(n_fft / 2 + 1,  // num_frequency_bins
                                             n_mels_,        // num_mel_filters
                                             0.0f,           // min_frequency
                                             8000.0f,        // max_frequency (for 16kHz audio)
                                             sample_rate     // sampling_rate
    );
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto audio = inputs[0];  // [B, T]

    // Handle padding
    if (padding_ == "same") {
      // int32_t pad = win_length_ - hop_length_;
      // audio = pad(audio, (pad / 2, pad / 2), "reflect");
      NYI("apply same padding in MelSpectrogramFeatures not implemented");
    }

    auto stft_result = stft_(audio, window_);

    // Compute magnitude spectrogram with power scaling
    // Whisper uses power=2 (energy spectrogram)
    auto specgram = stft_result.abs();
    if (power_ == 2) {
      specgram = specgram * specgram;  // Square the magnitude for power=2
    } else if (power_ != 1) {
      NYI("power != 1 and power != 2 not implemented");
    }

    // Apply mel filterbank: mel_spec = mel_filters.T @ specgram
    auto mel_specgram = nn::functional::matmul(specgram.T(), melscale_fbanks_).T();  // Use melscale_fbanks_ directly

    // Whisper-specific log mel processing:
    // 1. Clamp to avoid log(0)
    mel_specgram = nn::functional::clip(mel_specgram, 1e-10, std::numeric_limits<float>::max());

    // 2. Apply log10 using log(x) / log(10)
    mel_specgram = nn::functional::log(mel_specgram) / std::log(10.0f);

    // 3. Dynamic range compression: max(log_spec, log_spec.max() - 8.0)
    //    This is equivalent to clip(log_spec, log_spec.max() - 8.0, +inf)
    auto max_val = mel_specgram.max();  // Get scalar max value
    float threshold = max_val.item<float>() - 8.0f;
    mel_specgram = nn::functional::clip(mel_specgram, threshold, std::numeric_limits<float>::max());

    // 4. Normalize to approximately [-1, 1] range
    mel_specgram = (mel_specgram + 4.0f) / 4.0f;

    return {mel_specgram};
  }
};

class MiniCPMOAudioProcessor {
  MelSpectrogramFeatures mel_extractor_;
  int32_t sample_rate_;
  int32_t hop_length_;
  int32_t chunk_length_;  // Chunk length in seconds (default: 30)
  int32_t n_samples_;     // Max samples = chunk_length * sample_rate (default: 480000)

 public:
  MelSpectrogramFeatures& getMelExtractor() { return mel_extractor_; }
  explicit MiniCPMOAudioProcessor(int32_t sample_rate = 16000, int32_t n_mels = 80, int32_t hop_length = 160,
                                  int32_t chunk_length = 30)
      : mel_extractor_("feature_extractor.mel_spec", sample_rate, 400, hop_length, n_mels, "center", 2),
        sample_rate_(sample_rate),
        hop_length_(hop_length),
        chunk_length_(chunk_length),
        n_samples_(chunk_length * sample_rate) {}

  [[nodiscard]] std::string getAudioPlaceholder(int32_t audio_length, bool chunk_input = false,
                                                float chunk_length = 1.0f) const {
    const int32_t pool_step = 2;

    // Calculate feature length after mel-spectrogram extraction
    int32_t feature_lens = static_cast<int32_t>(std::ceil(static_cast<float>(audio_length) / hop_length_));

    // After first downsampling (Conv1D with stride 2)
    feature_lens = (feature_lens - 1) / 2 + 1;

    // After second pooling/downsampling
    int32_t output_lens = (feature_lens - pool_step) / pool_step + 1;

    std::string audio_placeholder;

    if (chunk_input) {
      // Calculate embeddings per chunk
      int32_t fbank_feat_in_chunk = static_cast<int32_t>(chunk_length * 100);
      int32_t cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) / 2 + 1;
      int32_t audio_embeds_in_chunk = (cnn_feat_in_chunk - pool_step) / pool_step + 1;
      int32_t num_audio_chunks = (output_lens + audio_embeds_in_chunk - 1) / audio_embeds_in_chunk;

      int32_t total_unk_len = 0;
      for (int32_t i = 0; i < num_audio_chunks; ++i) {
        int32_t unk_len = std::min(audio_embeds_in_chunk, output_lens - total_unk_len);
        audio_placeholder += "<|audio_start|>";
        for (int32_t j = 0; j < unk_len; ++j) { audio_placeholder += "<unk>"; }
        audio_placeholder += "<|audio_end|>";
        total_unk_len += unk_len;
      }
    } else {
      audio_placeholder = "<|audio_start|>";
      for (int32_t i = 0; i < output_lens; ++i) { audio_placeholder += "<unk>"; }
      audio_placeholder += "<|audio_end|>";
    }

    return audio_placeholder;
  }

  // Process audio file and extract mel-spectrogram features
  [[nodiscard]] Tensor processAudio(const std::string& audio_file_path) {
    // 1. Load audio file using existing mllm::audio::readWAV
    auto audio_data = mllm::audio::readWAV(audio_file_path, sample_rate_);

    if (audio_data.empty()) { return Tensor::nil(); }

    // 2. Pad or truncate audio to n_samples_ (30 seconds = 480000 samples @ 16kHz)
    // This matches Whisper's behavior of padding to chunk_length
    int32_t audio_length = static_cast<int32_t>(audio_data.size());
    int32_t padded_length = n_samples_;

    auto audio_tensor = Tensor::empty({1, padded_length}, kFloat32, kCPU).alloc();
    float* audio_ptr = audio_tensor.ptr<float>();

    if (audio_length <= padded_length) {
      // Copy audio data and pad with zeros
      std::copy(audio_data.begin(), audio_data.end(), audio_ptr);
      std::fill(audio_ptr + audio_length, audio_ptr + padded_length, 0.0f);
    } else {
      // Truncate to n_samples_
      std::copy(audio_data.begin(), audio_data.begin() + padded_length, audio_ptr);
    }

    // 3. Extract mel-spectrogram using WhisperMelSpectrogramFeatures
    // This will:
    //   - Compute STFT with center padding
    //   - Apply mel filterbank
    //   - Convert to log scale
    auto mel_spec = mel_extractor_.forward({audio_tensor}, {})[0];  // [1, n_mels, n_frames]

    return mel_spec;
  }

  // Process raw audio data and extract mel-spectrogram features
  [[nodiscard]] Tensor processAudioData(const float* audio_data, int32_t audio_length) {
    if (audio_data == nullptr || audio_length <= 0) { return Tensor::nil(); }

    // Pad or truncate audio to n_samples_ (30 seconds = 480000 samples @ 16kHz)
    // ***This matches Whisper's behavior of padding to chunk_length!!***
    int32_t padded_length = n_samples_;

    auto audio_tensor = Tensor::empty({1, padded_length}, kFloat32, kCPU).alloc();
    float* audio_ptr = audio_tensor.ptr<float>();

    if (audio_length <= padded_length) {
      // Copy audio data and pad with zeros
      std::memcpy(audio_ptr, audio_data, audio_length * sizeof(float));
      std::fill(audio_ptr + audio_length, audio_ptr + padded_length, 0.0f);
    } else {
      // Truncate to n_samples_
      std::memcpy(audio_ptr, audio_data, padded_length * sizeof(float));
    }

    // Extract mel-spectrogram
    auto mel_spec = mel_extractor_.forward({audio_tensor}, {})[0];  // [1, n_mels, n_frames]

    return mel_spec;
  }

  // Calculate audio bounds in token sequence
  // Similar to calculate_image_bounds in tokenization
  [[nodiscard]] std::vector<std::pair<int32_t, int32_t>> calcAudioBounds(const std::vector<int64_t>& input_ids,
                                                                         int64_t audio_start_id, int64_t audio_end_id) const {
    std::vector<std::pair<int32_t, int32_t>> audio_bounds;

    for (size_t i = 0; i < input_ids.size(); ++i) {
      if (input_ids[i] == audio_start_id) {
        // Find corresponding end token
        for (size_t j = i + 1; j < input_ids.size(); ++j) {
          if (input_ids[j] == audio_end_id) {
            // Store bounds (start + 1, end) to exclude the delimiter tokens
            audio_bounds.emplace_back(static_cast<int32_t>(i + 1), static_cast<int32_t>(j));
            break;
          }
        }
      }
    }

    return audio_bounds;
  }
};

}  // namespace mllm::models::minicpmo
