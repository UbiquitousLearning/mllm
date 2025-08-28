// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//

#include "Audio.hpp"
#include <iomanip>
#include <numeric>
#include <utility>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include "mllm/utils/Common.hpp"
#include <wenet_audio/params.h>
#include <wenet_audio/wav.h>
#include <wenet_audio/feature_pipeline.h>

class Fraction {
 public:
  int numerator;
  int denominator;

  // Constructor accepting a single int argument
  explicit Fraction(int num) : numerator(num), denominator(1) {}

  Fraction(int num, int den) : numerator(num), denominator(den) {
    if (denominator == 0) { throw std::invalid_argument("Denominator cannot be zero."); }
    simplify();
  }

  // Calculate the greatest common divisor
  int gcd(int a, int b) {
    while (b != 0) {
      int temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  }

  // Simplify the fraction
  void simplify() {
    if (numerator == 0) {
      denominator = 1;
      return;
    }
    int gcd_value = gcd(std::abs(numerator), std::abs(denominator));
    numerator /= gcd_value;
    denominator /= gcd_value;
    if (denominator < 0) {
      numerator = -numerator;
      denominator = -denominator;
    }
  }

  // Addition operation
  Fraction operator+(const Fraction& other) {
    int num = numerator * other.denominator + other.numerator * denominator;
    int den = denominator * other.denominator;
    return {num, den};
  }

  // Subtraction operation
  Fraction operator-(const Fraction& other) {
    int num = numerator * other.denominator - other.numerator * denominator;
    int den = denominator * other.denominator;
    return {num, den};
  }

  // Multiplication operation
  Fraction operator*(const Fraction& other) {
    int num = numerator * other.numerator;
    int den = denominator * other.denominator;
    return {num, den};
  }

  // Division operation
  Fraction operator/(const Fraction& other) {
    if (other.numerator == 0) { throw std::invalid_argument("Cannot divide by zero."); }
    int num = numerator * other.denominator;
    int den = denominator * other.numerator;
    return {num, den};
  }

  // Overload comparison operators
  bool operator>(const Fraction& other) const {
    return static_cast<double>(numerator) / denominator > static_cast<double>(other.numerator) / other.denominator;
  }

  bool operator<(const Fraction& other) const {
    return static_cast<double>(numerator) / denominator < static_cast<double>(other.numerator) / other.denominator;
  }

  bool operator>=(const Fraction& other) const { return !(*this < other); }

  bool operator<=(const Fraction& other) const { return !(*this > other); }

  bool operator==(const Fraction& other) const { return numerator == other.numerator && denominator == other.denominator; }

  bool operator!=(const Fraction& other) const { return !(*this == other); }

  // Print the fraction
  void print() { std::cout << numerator << "/" << denominator << '\n'; }

  [[nodiscard]] float toFloat() const { return static_cast<float>(numerator) / denominator; }

  static Fraction max(const Fraction& a, const Fraction& b) {
    double val_a = static_cast<double>(a.numerator) / a.denominator;
    double val_b = static_cast<double>(b.numerator) / b.denominator;
    if (val_a >= val_b) {
      return a;
    } else {
      return b;
    }
  }
};

namespace MLLM_ANONYMOUS_NAMESPACE {

float* waveClip(const float* data_, int start, int end, int channel) {
  std::vector<float> even_elements;

  for (int i = start; i < end; ++i) {
    if (i % channel == 0) { even_elements.push_back(data_[i]); }
  }
  auto data_new = new float[(end - start) / channel];
  std::copy(even_elements.begin(), even_elements.end(), data_new);
  return data_new;
}
std::vector<std::vector<float>> readFeats(const std::shared_ptr<wenet::FeaturePipeline>& feature_pipeline,
                                          const int num_frames_, const int feature_dim_) {
  bool end_flag = false;
  std::vector<std::vector<float>> chunk_feats;
  while (!end_flag) {
    // Read `num_frames_` of frame and extract features.
    if (!feature_pipeline->Read(num_frames_, &chunk_feats)) {
      // If the feat is end, pad the feat to `num_frames_` frames.
      int padding_len = num_frames_ - chunk_feats.size();
      std::vector<float> zero_vector(feature_dim_, 0);
      for (int i = 0; i < padding_len; i++) { chunk_feats.push_back(zero_vector); }
      end_flag = true;
    }
  }
  return chunk_feats;
}
std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> chunk_feats) {
  std::vector<std::vector<float>> transposed(chunk_feats[0].size(), std::vector<float>(chunk_feats.size()));
  for (size_t i = 0; i < chunk_feats.size(); ++i) {
    for (size_t j = 0; j < chunk_feats[i].size(); ++j) transposed[j][i] = chunk_feats[i][j];
  }
  return transposed;
}
void print2DVetcors(const std::vector<std::vector<float>>& chunk_feats) {
  std::cout << std::fixed;
  std::cout << std::setprecision(4);
  for (auto& chunk_feat : chunk_feats) {
    for (float j : chunk_feat) { std::cout << j << ","; }
    std::cout << '\n';
  }
}
void print3DVetcors(const std::vector<std::vector<std::vector<float>>>& all_clips) {
  for (const auto& all_clip : all_clips) {
    print2DVetcors(all_clip);
    std::cout << "======================================" << '\n';
  }
}

void Normalize(std::vector<std::vector<float>>& chunk_feats, const float mean, const float std) {
  for (auto& chunk_feat : chunk_feats) {
    for (float& j : chunk_feat) { j = (j - mean) / std; }
  }
}

void printdata_(const float* data_, int num_data) {
  std::cout << std::fixed;
  std::cout << std::setprecision(8);
  for (int i = 0; i < num_data; i++) { std::cout << data_[i] << " "; }
  std::cout << '\n';
  std::cout << num_data << '\n';
}

std::pair<int, std::vector<std::vector<std::vector<float>>>> get_sinc_resample_kernel(int new_freq, int gcd, int orig_freq) {
  // int orig_freq = sample_rate_;
  float lowpass_filter_width = 6;
  float rolloff = 0.99;
  std::string resampling_method = "sinc_interp_hann";

  orig_freq = int(orig_freq) / gcd;
  new_freq = int(new_freq) / gcd;

  if (lowpass_filter_width <= 0) std::cout << "lowpass_filter_width must be positive" << '\n';
  float base_freq = std::min(orig_freq, new_freq);
  base_freq *= rolloff;

  int width = ceil(lowpass_filter_width * orig_freq / base_freq);

  std::vector<float> idx(orig_freq + 2 * width);
  for (int i = 0; i < orig_freq + 2 * width; i++) { idx[i] = float(-width + i) / float(orig_freq); }

  std::vector<float> t(new_freq * idx.size());
  float t_temp = 0;
  for (int i = 0; i < new_freq; i++) {
    for (int j = 0; j < idx.size(); j++) { t[i * idx.size() + j] += t_temp; }
    t_temp -= 1;
  }

  for (float& i : t) { i = float(i) / float(new_freq); }

  // 将 t 和 idx 相加
  for (int i = 0; i < new_freq; i++) {
    for (int j = 0; j < idx.size(); j++) { t[i * idx.size() + j] += idx[j]; }
  }

  for (float& i : t) { i = float(i) * float(base_freq); }

  for (auto& value : t) { value = std::max(-lowpass_filter_width, std::min(value, lowpass_filter_width)); }

  std::vector<float> window(new_freq * idx.size());
  for (int i = 0; i < t.size(); i++) { window[i] = std::pow(cosf(t[i] * M_PI / lowpass_filter_width / 2), 2); }

  for (float& i : t) { i = i * M_PI; }

  float scale = base_freq / orig_freq;

  std::vector<float> kernels(t.size());
  std::transform(t.begin(), t.end(), kernels.begin(), [](double val) { return (val == 0) ? 1.0 : std::sin(val) / val; });

  std::vector<std::vector<std::vector<float>>> result;
  result.resize(new_freq);
  for (auto& res : result) {
    res.resize(1);
    res[0].resize(kernels.size() / new_freq);
  }
  for (int i = 0; i < kernels.size(); i++) {
    kernels[i] = kernels[i] * window[i] * scale;
    result[i / (kernels.size() / new_freq)][0][i % (kernels.size() / new_freq)] = kernels[i];
  }
  return std::make_pair(width, result);
}

std::vector<std::vector<float>> wav_pad(std::vector<std::vector<float>> orig_wav, int pad_left, int pad_right) {
  std::vector<std::vector<float>> result;
  result.resize(orig_wav.size());
  for (auto& re : result) { re.resize(orig_wav[0].size() + pad_left + pad_right); }
  for (int i = 0; i < orig_wav.size(); i++) {
    memset(result[i].data(), 0, pad_left * sizeof(float));
    memcpy(result[i].data() + pad_left, orig_wav[i].data(), orig_wav[i].size() * sizeof(float));
    memset(result[i].data() + pad_left + orig_wav[i].size(), 0, pad_right * sizeof(float));
  }
  return result;
}
std::vector<std::vector<std::vector<float>>> conv1d(std::vector<std::vector<float>> input,
                                                    std::vector<std::vector<std::vector<float>>> kernel, int stride) {
  int batch_size = input.size();
  int input_dim = input[0].size();
  int out_channels = kernel.size();
  int in_channels = kernel[0].size();
  int kernel_size = kernel[0][0].size();

  int output_dim = (input_dim - kernel_size) / stride + 1;

  std::vector<std::vector<std::vector<float>>> output(
      batch_size, std::vector<std::vector<float>>(out_channels, std::vector<float>(output_dim, 0.0)));

  // convolution
#pragma omp parallel for collapse(3) num_threads(4)
  for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int i = 0; i < output_dim; ++i) {
        // compute the convolution result
        float value = 0.0;
        for (int ic = 0; ic < in_channels; ++ic) {
          float tmp_value = 0.0;
          // TODO: SIMD accelerate
          for (int j = 0; j < kernel_size; ++j) { tmp_value += input[b][i * stride + j] * kernel[oc][ic][j]; }
          value += tmp_value;
        }
        output[b][oc][i] = value;
      }
    }
  }

  return output;
}

std::vector<std::vector<float>> conv1d_and_trans_and_viw(std::vector<std::vector<float>> input,
                                                         std::vector<std::vector<std::vector<float>>> kernel, int stride) {
  int batch_size = input.size();
  int input_dim = input[0].size();
  int out_channels = kernel.size();
  int in_channels = kernel[0].size();
  int kernel_size = kernel[0][0].size();

  int output_dim = (input_dim - kernel_size) / stride + 1;

  std::vector<std::vector<float>> output(batch_size, std::vector<float>(out_channels * output_dim, 0.0));

  // convolution
#pragma omp parallel for collapse(3) num_threads(4)
  for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int i = 0; i < output_dim; ++i) {
        // compute the convolution result
        float value = 0.0;
        for (int ic = 0; ic < in_channels; ++ic) {
          float tmp_value = 0.0;
          // TODO: SIMD accelerate
          for (int j = 0; j < kernel_size; ++j) { tmp_value += input[b][i * stride + j] * kernel[oc][ic][j]; }
          value += tmp_value;
        }
        output[b][i * out_channels + oc] = value;
      }
    }
  }

  return output;
}
std::vector<float> cut_and_trans(std::vector<std::vector<float>> wav, int target_length) {
  std::vector<float> result(target_length * wav.size());
  for (int i = 0; i < target_length * wav.size(); ++i) {
    auto dim_b = i % wav.size();
    auto dim_d = i / wav.size();
    result[i] = wav[dim_b][dim_d];
  }
  return result;
}
std::vector<float> apply_sinc_resample_kernel(std::vector<std::vector<float>> orig_wav, int orig_freq, int new_freq, int gcd,
                                              std::vector<std::vector<std::vector<float>>> kernel, int width) {
  auto length = orig_wav[0].size();
  orig_freq = int(orig_freq) / gcd;
  new_freq = int(new_freq) / gcd;
  std::vector<float> result;
  orig_wav = wav_pad(orig_wav, width, width + orig_freq);
  auto resample_wav = conv1d_and_trans_and_viw(orig_wav, std::move(kernel), orig_freq);

  int target_length = static_cast<int>(std::ceil(static_cast<double>(new_freq) * length / orig_freq));
  result = cut_and_trans(resample_wav, target_length);
  return result;
}

std::vector<float> resample(std::vector<std::vector<float>> orig_wav, int new_freq, int orig_freq) {
  std::vector<float> resampled;
  // int orig_freq = sample_rate_;
  if (new_freq == orig_freq) { return resampled; }
  int gcd = std::gcd(new_freq, orig_freq);

  auto width_kernel = get_sinc_resample_kernel(new_freq, gcd, orig_freq);
  int width = width_kernel.first;
  auto kernel = width_kernel.second;

  resampled = apply_sinc_resample_kernel(std::move(orig_wav), orig_freq, new_freq, gcd, kernel, width);
  return resampled;
}

std::vector<std::vector<float>> get_wav_data(const float* wavdata, int wavdata_sample, int wavdata_channel) {
  std::vector<std::vector<float>> result;
  result.resize(wavdata_channel);
  for (auto& data : result) { data.resize(wavdata_sample); }
  for (int i = 0; i < wavdata_sample * wavdata_channel; i++) { result[i % wavdata_channel][i / wavdata_channel] = wavdata[i]; }

  return result;
}

int current_aug_index = 0;
int current_clip_index = 0;
bool is_last_clip = false;

Fraction clip_sampler(float last_clip_end_time, Fraction video_duration, Fraction clip_duration, Fraction clips_per_video) {
  int augs_per_clip = 1;

  auto max_possible_clip_start = Fraction::max(Fraction(0), video_duration - clip_duration);

  auto uniform_clip = max_possible_clip_start / Fraction::max(clips_per_video - Fraction(1), Fraction(1));

  auto clip_start_sec = uniform_clip * Fraction(current_clip_index);
  int clip_index = current_clip_index;
  int aug_index = current_aug_index;

  current_aug_index += 1;
  if (current_aug_index >= augs_per_clip) {
    current_clip_index += 1;
    current_aug_index = 0;
  }

  if (Fraction(current_clip_index) >= clips_per_video
      || uniform_clip * Fraction(current_clip_index) > max_possible_clip_start) {
    current_clip_index = 0;
    is_last_clip = true;
  }

  if (is_last_clip) { current_clip_index = 0; }

  return clip_start_sec;
}

std::vector<std::pair<Fraction, Fraction>> get_clip_timepoints(Fraction clip_duration, Fraction clips_per_video,
                                                               Fraction duration) {
  std::vector<std::pair<Fraction, Fraction>> all_clip_timepoints;
  float end = 0;
  Fraction clip_sampler_result = clip_sampler(end, duration, clip_duration, clips_per_video);
  all_clip_timepoints.emplace_back(clip_sampler_result, clip_sampler_result + clip_duration);
  while (is_last_clip == 0) {
    clip_sampler_result = clip_sampler(end, duration, clip_duration, clips_per_video);
    all_clip_timepoints.emplace_back(clip_sampler_result, clip_sampler_result + clip_duration);
  }
  return all_clip_timepoints;
}
std::vector<std::pair<int, int>> get_clip_timepoints(Fraction clip_duration, Fraction clips_per_video, Fraction duration,
                                                     int resample_rate) {
  current_aug_index = 0;
  current_clip_index = 0;
  is_last_clip = false;
  std::vector<std::pair<Fraction, Fraction>> clip_timepoints_test =
      get_clip_timepoints(clip_duration, clips_per_video, duration);
  std::vector<std::pair<int, int>> clip_timepoints;
  for (auto& values : clip_timepoints_test) {
    values.first = values.first * Fraction(resample_rate);
    values.second = values.second * Fraction(resample_rate);
    clip_timepoints.emplace_back(int(values.first.toFloat()), int(values.second.toFloat()));
  }
  return clip_timepoints;
}

// Convert stereo audio data to mono by averaging channels
std::vector<float> toMono(const std::vector<std::vector<float>>& wav_data) {
  if (wav_data.size() == 1) { return wav_data[0]; }

  size_t num_samples = wav_data[0].size();
  std::vector<float> mono_data(num_samples);

  for (size_t i = 0; i < num_samples; ++i) {
    float sum = 0.0;
    for (const auto& ch : wav_data) { sum += ch[i]; }
    mono_data[i] = sum / wav_data.size();
  }

  return mono_data;
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

namespace mllm::audio {
std::vector<std::vector<std::vector<std::vector<float>>>> processWAV(const std::vector<std::string>& waves, int resample_rate) {
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto feature_pipeline = std::make_shared<wenet::FeaturePipeline>(*feature_config);
  std::vector<std::vector<std::vector<std::vector<float>>>> output_audios;
  Fraction clip_duration(2);
  Fraction clips_per_video(3);
  for (auto& wav : waves) {
    wenet::WavReader wav_reader(wav);
    wav_reader.rescale();
    auto wavdata = wav_reader.data();
    auto wavdata_sample = wav_reader.num_sample();
    auto wavdata_channel = wav_reader.num_channel();
    auto origin_sample_rate = wav_reader.sample_rate();
    auto wavdata_size = wavdata_sample * wavdata_channel;
    std::vector<std::vector<float>> wav_data = get_wav_data(wavdata, wavdata_sample, wavdata_channel);
    auto resampled = resample(wav_data, resample_rate, origin_sample_rate);

    auto waveform_size = wav_reader.num_sample();
    if (origin_sample_rate != resample_rate) { waveform_size = resampled.size() / wavdata_channel; }
    std::vector<std::pair<int, int>> clip_timepoints = get_clip_timepoints(
        Fraction(clip_duration), Fraction(clips_per_video), Fraction(waveform_size) / Fraction(resample_rate), resample_rate);
    std::vector<std::vector<std::vector<float>>> all_clips;
    for (auto clip_timepoint : clip_timepoints) {
      const int clip_start = clip_timepoint.first * wav_reader.num_channel();
      const int clip_end = clip_timepoint.second * wav_reader.num_channel();
      const float* datac;
      if (origin_sample_rate == resample_rate) {
        datac = waveClip(wav_reader.data(), clip_start, clip_end, wav_reader.num_channel());
      } else {
        datac = waveClip(resampled.data(), clip_start, clip_end, wav_reader.num_channel());
      }
      const int datac_num_sample = (clip_end - clip_start) / wav_reader.num_channel();
      feature_pipeline->AcceptWaveform(std::vector<float>(datac, datac + datac_num_sample));
      const int num_frames_ = 204;
      const int feature_dim_ = 128;
      const auto chunk_feats = readFeats(feature_pipeline, num_frames_, feature_dim_);
      auto outfeats = transpose(chunk_feats);
      Normalize(outfeats, -4.268, 9.138);
      all_clips.push_back(outfeats);
    }
    // print3DVetcors(all_clips);
    output_audios.push_back(all_clips);
  }
  return output_audios;
}

std::vector<float> readWAV(const std::string& file_path, int resample_rate) {
  // Read WAV file using wenet library
  wenet::WavReader wav_reader(file_path);
  wav_reader.rescale();

  // Get audio properties
  auto wavdata = wav_reader.data();
  auto wavdata_sample = wav_reader.num_sample();
  auto wavdata_channel = wav_reader.num_channel();
  auto origin_sample_rate = wav_reader.sample_rate();

  // Convert raw data to channel-wise data
  std::vector<std::vector<float>> wav_data = get_wav_data(wavdata, wavdata_sample, wavdata_channel);

  // Resample
  std::vector<float> resampled_data;
  if (origin_sample_rate != resample_rate) { resampled_data = resample(wav_data, resample_rate, origin_sample_rate); }

  // Convert to mono
  std::vector<float> mono_data;
  if (resampled_data.empty()) {
    // No resampling was needed, use original data
    mono_data = toMono(wav_data);
  } else {
    // Resampling was performed, convert resampled data to channel-wise format first
    size_t num_channels = wav_data.size();
    size_t resampled_samples = resampled_data.size() / num_channels;
    std::vector<std::vector<float>> resampled_wav_data(num_channels, std::vector<float>(resampled_samples));
    for (size_t i = 0; i < resampled_data.size(); ++i) {
      resampled_wav_data[i % num_channels][i / num_channels] = resampled_data[i];
    }
    mono_data = toMono(resampled_wav_data);
  }

  return mono_data;
}

void writeWAV(const std::vector<float>& data, int sample_rate, int num_channels, const std::string& file_path) {
  // Create WavWriter with the provided data
  wenet::WavWriter wav_writer(data.data(), data.size() / num_channels, num_channels, sample_rate, 16);
  wav_writer.Write(file_path);
}

}  // namespace mllm::audio