// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <cmath>
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/preprocessor/audio/Audio.hpp"

namespace mllm::models::minicpmo {

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
 * Audio preprocessor for MiniCPM-O
 * Converts audio files to mel-spectrogram features compatible with Whisper encoder
 */
class MiniCPMOAudioProcessor {
  MelSpectrogramFeatures mel_extractor_;
  int32_t sample_rate_;
  int32_t hop_length_;

 public:
  MelSpectrogramFeatures& getMelExtractor() { return mel_extractor_; }
  explicit MiniCPMOAudioProcessor(int32_t sample_rate = 16000, int32_t n_mels = 100, int32_t hop_length = 160)
      : mel_extractor_("feature_extractor.mel_spec", sample_rate, 400, hop_length, n_mels),
        sample_rate_(sample_rate),
        hop_length_(hop_length) {}

  /**
   * Calculate audio placeholder based on audio length
   * @param audio_length Audio length in samples
   * @param chunk_input Whether to use chunked input
   * @param chunk_length Chunk length in seconds (default 1.0)
   * @return Audio placeholder string
   */
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

  /**
   * Process audio file and extract mel-spectrogram features
   * @param audio_file_path Path to audio file
   * @return Tensor containing mel-spectrogram features [batch=1, num_mel_bins, sequence_length]
   */
  [[nodiscard]] Tensor processAudio(const std::string& audio_file_path) {
    // 1. Load audio file using existing mllm::audio::readWAV
    auto audio_data = mllm::audio::readWAV(audio_file_path, sample_rate_);

    if (audio_data.empty()) { return Tensor::nil(); }

    // 2. Convert to Tensor [1, T]
    auto audio_tensor = Tensor::empty({1, static_cast<int32_t>(audio_data.size())}, kFloat32, kCPU).alloc();
    std::copy(audio_data.begin(), audio_data.end(), audio_tensor.ptr<float>());

    // 3. Extract mel-spectrogram using WhisperMelSpectrogramFeatures
    // This will:
    //   - Compute STFT with center padding
    //   - Apply mel filterbank
    //   - Convert to log scale
    auto mel_spec = mel_extractor_.forward({audio_tensor}, {})[0];  // [1, n_mels, n_frames]

    return mel_spec;
  }

  /**
   * Process raw audio data and extract mel-spectrogram features
   * @param audio_data Raw audio samples (mono, float)
   * @param audio_length Length of audio data
   * @return Tensor containing mel-spectrogram features [batch=1, num_mel_bins, sequence_length]
   */
  [[nodiscard]] Tensor processAudioData(const float* audio_data, int32_t audio_length) {
    if (audio_data == nullptr || audio_length <= 0) { return Tensor::nil(); }

    // Convert to Tensor [1, T]
    auto audio_tensor = Tensor::empty({1, audio_length}, kFloat32, kCPU).alloc();
    std::copy(audio_data, audio_data + audio_length, audio_tensor.ptr<float>());

    // Extract mel-spectrogram
    auto mel_spec = mel_extractor_.forward({audio_tensor}, {})[0];  // [1, n_mels, n_frames]

    return mel_spec;
  }

  /**
   * Load mel filter parameters from pretrained model
   * @param param ParameterFile containing mel filter weights
   */
  void loadMelFilters(const mllm::ParameterFile::ptr_t& param) { mel_extractor_.load(param); }

  /**
   * Calculate audio bounds in token sequence
   * Similar to image_bounds calculation
   * @param input_ids Token IDs
   * @param audio_start_id Token ID for <|audio_start|>
   * @param audio_end_id Token ID for <|audio_end|>
   * @return Vector of (start, end) pairs indicating audio token positions
   */
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
