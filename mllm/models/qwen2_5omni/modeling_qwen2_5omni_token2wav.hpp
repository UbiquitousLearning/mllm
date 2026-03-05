// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "mllm/core/Parallel.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Enumerate.hpp"

#include "mllm/models/qwen2_5omni/configuration_qwen2_5omni.hpp"

namespace mllm::models::qwen2_5omni {

namespace token2wav {

constexpr float kPi = 3.14159265358979323846f;

inline Tensor pad1dReflect(const Tensor& x, int32_t pad_left, int32_t pad_right) {
  if (pad_left == 0 && pad_right == 0) { return x; }
  return nn::functional::pad(x, {pad_left, pad_right}, aops::PadMode::kReflect);
}

inline Tensor pad1dReplicate(const Tensor& x, int32_t pad_left, int32_t pad_right) {
  if (pad_left == 0 && pad_right == 0) { return x; }
  return nn::functional::pad(x, {pad_left, pad_right}, aops::PadMode::kReplicate);
}

inline Tensor clampTensor(const Tensor& x, float min_val, float max_val) {
  MLLM_RT_ASSERT_EQ(x.device(), kCPU);
  MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);

  auto out = Tensor::empty(x.shape(), x.dtype(), x.device()).alloc();
  const auto* src = x.ptr<float>();
  auto* dst = out.ptr<float>();
  const auto numel = x.numel();

  MLLM_CONDITIONAL_PARALLEL_FOR(numel > 1024, 4, idx, 0, numel, 1, {
    float v = src[idx];
    v = std::min(std::max(v, min_val), max_val);
    dst[idx] = v;
  });
  return out;
}

inline Tensor amplitudeToDb(const Tensor& amplitude, float min_db_level) {
  MLLM_RT_ASSERT_EQ(amplitude.device(), kCPU);
  MLLM_RT_ASSERT_EQ(amplitude.dtype(), kFloat32);

  const float min_level = std::exp(min_db_level / 20.0f * std::log(10.0f));
  const float log10_scale = 1.0f / std::log(10.0f);

  auto out = Tensor::empty(amplitude.shape(), amplitude.dtype(), amplitude.device()).alloc();
  const auto* src = amplitude.ptr<float>();
  auto* dst = out.ptr<float>();
  const auto numel = amplitude.numel();

  MLLM_CONDITIONAL_PARALLEL_FOR(numel > 1024, 4, idx, 0, numel, 1, {
    float v = std::max(src[idx], min_level);
    dst[idx] = 20.0f * std::log(v) * log10_scale;
  });

  return out;
}

inline Tensor normalizeSpectrogram(const Tensor& spectrogram, float max_value, float min_db) {
  MLLM_RT_ASSERT_EQ(spectrogram.device(), kCPU);
  MLLM_RT_ASSERT_EQ(spectrogram.dtype(), kFloat32);

  auto out = Tensor::empty(spectrogram.shape(), spectrogram.dtype(), spectrogram.device()).alloc();
  const auto* src = spectrogram.ptr<float>();
  auto* dst = out.ptr<float>();
  const auto numel = spectrogram.numel();

  const float scale = (2.0f * max_value) / (-min_db);
  MLLM_CONDITIONAL_PARALLEL_FOR(numel > 1024, 4, idx, 0, numel, 1, {
    float v = scale * (src[idx] - min_db) - max_value;
    v = std::min(std::max(v, -max_value), max_value);
    dst[idx] = v;
  });
  return out;
}

inline float besselI0(float x) {
  const float ax = std::abs(x);
  if (ax < 3.75f) {
    const float y = (ax / 3.75f);
    const float y2 = y * y;
    return 1.0f + y2 * (3.5156229f +
                        y2 * (3.0899424f +
                              y2 * (1.2067492f +
                                    y2 * (0.2659732f +
                                          y2 * (0.0360768f +
                                                y2 * 0.0045813f)))));
  }

  const float y = 3.75f / ax;
  const float exp_ax = std::exp(ax);
  return (exp_ax / std::sqrt(ax)) *
         (0.39894228f +
          y * (0.01328592f +
               y * (0.00225319f +
                    y * (-0.00157565f +
                         y * (0.00916281f +
                              y * (-0.02057706f +
                                   y * (0.02635537f +
                                        y * (-0.01647633f +
                                             y * 0.00392377f))))))));
}

inline Tensor kaiserSincFilter1d(float cutoff, float half_width, int32_t kernel_size) {
  const bool is_even = (kernel_size % 2) == 0;
  const int32_t half_size = kernel_size / 2;

  if (cutoff == 0.0f) { return Tensor::zeros({1, 1, kernel_size}, kFloat32, kCPU); }

  const float delta_f = 4.0f * half_width;
  const float attenuation = 2.285f * static_cast<float>(half_size - 1) * kPi * delta_f + 7.95f;

  float beta = 0.0f;
  if (attenuation > 50.0f) {
    beta = 0.1102f * (attenuation - 8.7f);
  } else if (attenuation >= 21.0f) {
    beta = 0.5842f * std::pow(attenuation - 21.0f, 0.4f) + 0.07886f * (attenuation - 21.0f);
  }

  const float denom = besselI0(beta);
  std::vector<float> window(kernel_size, 1.0f);
  for (int32_t n = 0; n < kernel_size; ++n) {
    const float ratio = (2.0f * static_cast<float>(n) / static_cast<float>(kernel_size - 1)) - 1.0f;
    const float val = std::sqrt(std::max(0.0f, 1.0f - ratio * ratio));
    window[n] = besselI0(beta * val) / denom;
  }

  std::vector<float> filter(kernel_size, 0.0f);
  float sum = 0.0f;
  for (int32_t n = 0; n < kernel_size; ++n) {
    float t = static_cast<float>(n) - static_cast<float>(half_size);
    if (is_even) { t += 0.5f; }
    const float arg = 2.0f * cutoff * t;
    const float sinc = (arg == 0.0f) ? 1.0f : std::sin(kPi * arg) / (kPi * arg);
    const float v = 2.0f * cutoff * window[n] * sinc;
    filter[n] = v;
    sum += v;
  }

  if (sum != 0.0f) {
    for (auto& v : filter) { v /= sum; }
  }

  auto out = Tensor::empty({1, 1, kernel_size}, kFloat32, kCPU).alloc();
  std::copy(filter.begin(), filter.end(), out.ptr<float>());
  return out;
}

inline Tensor convTranspose1dDepthwise(const Tensor& input, const Tensor& filter, int32_t stride) {
  MLLM_RT_ASSERT_EQ(input.device(), kCPU);
  MLLM_RT_ASSERT_EQ(input.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(filter.device(), kCPU);
  MLLM_RT_ASSERT_EQ(filter.dtype(), kFloat32);

  const auto& in_shape = input.shape();
  const int32_t batch = in_shape[0];
  const int32_t channels = in_shape[1];
  const int32_t in_len = in_shape[2];
  const int32_t kernel = filter.shape()[2];

  const int32_t out_len = (in_len - 1) * stride + kernel;
  auto out = Tensor::zeros({batch, channels, out_len}, kFloat32, kCPU);

  const auto* in_ptr = input.ptr<float>();
  const auto* filt_ptr = filter.ptr<float>();
  auto* out_ptr = out.ptr<float>();

  const int32_t in_step = channels * in_len;
  const int32_t out_step = channels * out_len;

  for (int32_t b = 0; b < batch; ++b) {
    const float* in_b = in_ptr + b * in_step;
    float* out_b = out_ptr + b * out_step;
    for (int32_t c = 0; c < channels; ++c) {
      const float* in_c = in_b + c * in_len;
      float* out_c = out_b + c * out_len;
      const float* f = filt_ptr;
      for (int32_t i = 0; i < in_len; ++i) {
        const float v = in_c[i];
        const int32_t base = i * stride;
        for (int32_t k = 0; k < kernel; ++k) { out_c[base + k] += v * f[k]; }
      }
    }
  }

  return out;
}

inline Tensor conv1dDepthwise(const Tensor& input, const Tensor& filter, int32_t stride) {
  MLLM_RT_ASSERT_EQ(input.device(), kCPU);
  MLLM_RT_ASSERT_EQ(input.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(filter.device(), kCPU);
  MLLM_RT_ASSERT_EQ(filter.dtype(), kFloat32);

  const auto& in_shape = input.shape();
  const int32_t batch = in_shape[0];
  const int32_t channels = in_shape[1];
  const int32_t in_len = in_shape[2];
  const int32_t kernel = filter.shape()[2];

  const int32_t out_len = (in_len - kernel) / stride + 1;
  auto out = Tensor::zeros({batch, channels, out_len}, kFloat32, kCPU);

  const auto* in_ptr = input.ptr<float>();
  const auto* filt_ptr = filter.ptr<float>();
  auto* out_ptr = out.ptr<float>();

  const int32_t in_step = channels * in_len;
  const int32_t out_step = channels * out_len;

  for (int32_t b = 0; b < batch; ++b) {
    const float* in_b = in_ptr + b * in_step;
    float* out_b = out_ptr + b * out_step;
    for (int32_t c = 0; c < channels; ++c) {
      const float* in_c = in_b + c * in_len;
      float* out_c = out_b + c * out_len;
      const float* f = filt_ptr;
      for (int32_t o = 0; o < out_len; ++o) {
        float sum = 0.0f;
        const int32_t base = o * stride;
        for (int32_t k = 0; k < kernel; ++k) { sum += in_c[base + k] * f[k]; }
        out_c[o] = sum;
      }
    }
  }

  return out;
}

inline Tensor randomNormal(const std::vector<int32_t>& shape, float mean = 0.0f, float std = 1.0f) {
  auto out = Tensor::empty(shape, kFloat32, kCPU).alloc();
  auto* ptr = out.ptr<float>();
  const int64_t numel = out.numel();
  std::mt19937 gen(static_cast<uint32_t>(mllm::Context::instance().getRandomState()));
  std::normal_distribution<float> dist(mean, std);
  for (int64_t i = 0; i < numel; ++i) { ptr[i] = dist(gen); }
  return out;
}

inline Tensor linspace(float start, float end, int32_t steps) {
  auto out = Tensor::empty({steps}, kFloat32, kCPU).alloc();
  auto* ptr = out.ptr<float>();
  if (steps <= 1) {
    if (steps == 1) { ptr[0] = start; }
    return out;
  }
  const float step = (end - start) / static_cast<float>(steps - 1);
  for (int32_t i = 0; i < steps; ++i) { ptr[i] = start + step * static_cast<float>(i); }
  return out;
}

inline Tensor repeatInterleave(const Tensor& input, int32_t repeats, int32_t dim) {
  MLLM_RT_ASSERT_EQ(input.device(), kCPU);
  MLLM_RT_ASSERT_EQ(input.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(dim, 1);

  if (repeats == 1) { return input; }

  const auto& shape = input.shape();
  const int32_t batch = shape[0];
  const int32_t seq_len = shape[1];
  const int32_t channels = shape[2];

  auto out = Tensor::empty({batch, seq_len * repeats, channels}, kFloat32, kCPU).alloc();
  const auto* src = input.ptr<float>();
  auto* dst = out.ptr<float>();

  const int64_t in_stride_b = static_cast<int64_t>(seq_len) * channels;
  const int64_t out_stride_b = static_cast<int64_t>(seq_len) * repeats * channels;

  for (int32_t b = 0; b < batch; ++b) {
    const float* src_b = src + b * in_stride_b;
    float* dst_b = dst + b * out_stride_b;
    for (int32_t s = 0; s < seq_len; ++s) {
      const float* src_s = src_b + static_cast<int64_t>(s) * channels;
      for (int32_t r = 0; r < repeats; ++r) {
        float* dst_s = dst_b + (static_cast<int64_t>(s) * repeats + r) * channels;
        std::memcpy(dst_s, src_s, sizeof(float) * channels);
      }
    }
  }

  return out;
}

class SnakeBeta final : public nn::Module {
  nn::Param alpha_;
  nn::Param beta_;
  float no_div_by_zero_ = 1e-9f;

 public:
  SnakeBeta() = default;
  SnakeBeta(const std::string& name, int32_t in_features) : nn::Module(name) {
    alpha_ = reg<nn::Param>("alpha", getModuleName() + ".alpha", std::vector<int32_t>{in_features});
    beta_ = reg<nn::Param>("beta", getModuleName() + ".beta", std::vector<int32_t>{in_features});
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = inputs[0];
    MLLM_RT_ASSERT_EQ(x.device(), kCPU);
    MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);
    if (!x.isContiguous()) { x = x.contiguous(); }

    const auto& shape = x.shape();
    const int32_t batch = shape[0];
    const int32_t channels = shape[1];
    const int32_t seq_len = shape[2];

    auto y = Tensor::empty(shape, kFloat32, kCPU).alloc();
    const auto* x_ptr = x.ptr<float>();
    auto* y_ptr = y.ptr<float>();

    auto alpha = alpha_.weight();
    auto beta = beta_.weight();
    const auto* alpha_ptr = alpha.ptr<float>();
    const auto* beta_ptr = beta.ptr<float>();

    const int32_t stride_c = seq_len;
    const int32_t stride_b = channels * seq_len;

    for (int32_t b = 0; b < batch; ++b) {
      for (int32_t c = 0; c < channels; ++c) {
        const float a = std::exp(alpha_ptr[c]);
        const float bb = std::exp(beta_ptr[c]);
        const float inv_b = 1.0f / (bb + no_div_by_zero_);
        const int32_t base = b * stride_b + c * stride_c;
        for (int32_t t = 0; t < seq_len; ++t) {
          float v = x_ptr[base + t];
          const float s = std::sin(v * a);
          v = v + inv_b * (s * s);
          y_ptr[base + t] = v;
        }
      }
    }

    return {y};
  }

};

class TorchActivation1d final : public nn::Module {
 public:
  TorchActivation1d() = default;
  TorchActivation1d(const std::string& name, int32_t channels, int32_t up_ratio = 2, int32_t down_ratio = 2,
                    int32_t up_kernel_size = 12, int32_t down_kernel_size = 12)
      : nn::Module(name),
        up_ratio_(up_ratio),
        down_ratio_(down_ratio),
        up_kernel_size_(up_kernel_size),
        down_kernel_size_(down_kernel_size) {
    act_ = reg<SnakeBeta>("act", channels);

    up_kernel_size_ = (up_kernel_size_ <= 0) ? static_cast<int32_t>(int(6 * up_ratio_ / 2) * 2) : up_kernel_size_;
    up_stride_ = up_ratio_;
    up_pad_ = up_kernel_size_ / up_ratio_ - 1;
    up_pad_left_ = up_pad_ * up_stride_ + (up_kernel_size_ - up_stride_) / 2;
    up_pad_right_ = up_pad_ * up_stride_ + (up_kernel_size_ - up_stride_ + 1) / 2;

    down_kernel_size_ = (down_kernel_size_ <= 0) ? static_cast<int32_t>(int(6 * down_ratio_ / 2) * 2) : down_kernel_size_;
    down_stride_ = down_ratio_;
    down_even_ = (down_kernel_size_ % 2) == 0;
    down_pad_left_ = down_kernel_size_ / 2 - (down_even_ ? 1 : 0);
    down_pad_right_ = down_kernel_size_ / 2;

    up_filter_ = kaiserSincFilter1d(0.5f / static_cast<float>(up_ratio_), 0.6f / static_cast<float>(up_ratio_), up_kernel_size_);
    down_filter_ =
        kaiserSincFilter1d(0.5f / static_cast<float>(down_ratio_), 0.6f / static_cast<float>(down_ratio_), down_kernel_size_);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = inputs[0];
    x = upsample(x);
    x = act_(x)[0];
    x = downsample(x);
    return {x};
  }

 private:
  Tensor upsample(const Tensor& input) const {
    auto padded = pad1dReplicate(input, up_pad_, up_pad_);
    auto out = convTranspose1dDepthwise(padded, up_filter_, up_stride_);
    out = out * static_cast<float>(up_ratio_);
    if (up_pad_left_ > 0 || up_pad_right_ > 0) {
      auto length = out.shape()[2];
      auto start = up_pad_left_;
      auto end = length - up_pad_right_;
      out = out[{kAll, kAll, {start, end}}];
    }
    return out;
  }

  Tensor downsample(const Tensor& input) const {
    auto padded = pad1dReplicate(input, down_pad_left_, down_pad_right_);
    auto out = conv1dDepthwise(padded, down_filter_, down_stride_);
    return out;
  }

  SnakeBeta act_;
  int32_t up_ratio_ = 2;
  int32_t down_ratio_ = 2;
  int32_t up_kernel_size_ = 12;
  int32_t down_kernel_size_ = 12;
  int32_t up_stride_ = 2;
  int32_t down_stride_ = 2;
  int32_t up_pad_ = 0;
  int32_t up_pad_left_ = 0;
  int32_t up_pad_right_ = 0;
  int32_t down_pad_left_ = 0;
  int32_t down_pad_right_ = 0;
  bool down_even_ = false;
  Tensor up_filter_ = Tensor::nil();
  Tensor down_filter_ = Tensor::nil();
};

class TimeDelayNetBlock final : public nn::Module {
 public:
  TimeDelayNetBlock() = default;
  TimeDelayNetBlock(const std::string& name, int32_t in_channels, int32_t out_channels, int32_t kernel_size, int32_t dilation)
      : nn::Module(name), kernel_size_(kernel_size), dilation_(dilation) {
    conv_ = reg<nn::Conv1D>("conv", in_channels, out_channels, kernel_size_, 1, 0, dilation_, 1, true);
    relu_ = reg<nn::ReLU>("relu");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = inputs[0];
    const int32_t pad_total = dilation_ * (kernel_size_ - 1);
    const int32_t pad_left = pad_total / 2;
    const int32_t pad_right = pad_total - pad_left;
    if (pad_total > 0) { x = pad1dReflect(x, pad_left, pad_right); }
    x = conv_(x);
    x = relu_(x);
    return {x};
  }

 private:
  nn::Conv1D conv_;
  nn::ReLU relu_;
  int32_t kernel_size_ = 1;
  int32_t dilation_ = 1;
};

class Res2NetBlock final : public nn::Module {
 public:
  Res2NetBlock() = default;
  Res2NetBlock(const std::string& name, int32_t in_channels, int32_t out_channels, int32_t scale, int32_t kernel_size, int32_t dilation)
      : nn::Module(name), scale_(scale) {
    const int32_t in_channel = in_channels / scale;
    const int32_t hidden_channel = out_channels / scale;
    blocks_ = reg<nn::ModuleList<TimeDelayNetBlock>>("blocks", scale_ - 1, in_channel, hidden_channel, kernel_size, dilation);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = inputs[0];
    const int32_t channels = x.shape()[1];
    const int32_t split = channels / scale_;

    std::vector<Tensor> outputs;
    outputs.reserve(scale_);
    Tensor output_part = Tensor::nil();

    for (int32_t i = 0; i < scale_; ++i) {
      auto hidden_part = x[{kAll, {i * split, (i + 1) * split}, kAll}];
      if (i == 0) {
        output_part = hidden_part;
      } else if (i == 1) {
        output_part = blocks_.list()[i - 1](hidden_part)[0];
      } else {
        output_part = blocks_.list()[i - 1](hidden_part + output_part)[0];
      }
      outputs.push_back(output_part);
    }

    auto out = nn::functional::concat(outputs, 1);
    return {out};
  }

 private:
  int32_t scale_ = 1;
  nn::ModuleList<TimeDelayNetBlock> blocks_;
};

class SqueezeExcitationBlock final : public nn::Module {
 public:
  SqueezeExcitationBlock() = default;
  SqueezeExcitationBlock(const std::string& name, int32_t in_channels, int32_t se_channels, int32_t out_channels)
      : nn::Module(name) {
    conv1_ = reg<nn::Conv1D>("conv1", in_channels, se_channels, 1, 1, 0, 1, 1, true);
    conv2_ = reg<nn::Conv1D>("conv2", se_channels, out_channels, 1, 1, 0, 1, 1, true);
    relu_ = reg<nn::ReLU>("relu");
    sigmoid_ = reg<nn::Sigmoid>("sigmoid");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto hidden_states = inputs[0];
    auto hidden_mean = nn::functional::mean(hidden_states, 2, true);
    hidden_mean = relu_(conv1_(hidden_mean));
    hidden_mean = sigmoid_(conv2_(hidden_mean));
    hidden_states = hidden_states * hidden_mean;
    return {hidden_states};
  }

 private:
  nn::Conv1D conv1_;
  nn::Conv1D conv2_;
  nn::ReLU relu_;
  nn::Sigmoid sigmoid_;
};

class AttentiveStatisticsPooling final : public nn::Module {
 public:
  AttentiveStatisticsPooling() = default;
  AttentiveStatisticsPooling(const std::string& name, int32_t channels, int32_t attention_channels)
      : nn::Module(name), channels_(channels) {
    tdnn_ = reg<TimeDelayNetBlock>("tdnn", channels * 3, attention_channels, 1, 1);
    tanh_ = reg<nn::Tanh>("tanh");
    conv_ = reg<nn::Conv1D>("conv", attention_channels, channels, 1, 1, 0, 1, 1, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto hidden_states = inputs[0];
    MLLM_RT_ASSERT_EQ(hidden_states.dtype(), kFloat32);
    MLLM_RT_ASSERT_EQ(hidden_states.device(), kCPU);

    const int32_t batch = hidden_states.shape()[0];
    const int32_t channels = hidden_states.shape()[1];
    const int32_t seq_len = hidden_states.shape()[2];

    auto mean = Tensor::empty({batch, channels}, kFloat32, kCPU).alloc();
    auto std = Tensor::empty({batch, channels}, kFloat32, kCPU).alloc();

    const auto* x_ptr = hidden_states.ptr<float>();
    auto* mean_ptr = mean.ptr<float>();
    auto* std_ptr = std.ptr<float>();

    const int32_t stride_c = seq_len;
    const int32_t stride_b = channels * seq_len;

    for (int32_t b = 0; b < batch; ++b) {
      for (int32_t c = 0; c < channels; ++c) {
        const int32_t base = b * stride_b + c * stride_c;
        float sum = 0.0f;
        for (int32_t t = 0; t < seq_len; ++t) { sum += x_ptr[base + t]; }
        float m = sum / static_cast<float>(seq_len);
        mean_ptr[b * channels + c] = m;

        float var = 0.0f;
        for (int32_t t = 0; t < seq_len; ++t) {
          float diff = x_ptr[base + t] - m;
          var += diff * diff;
        }
        var /= static_cast<float>(seq_len);
        std_ptr[b * channels + c] = std::sqrt(std::max(var, 1e-12f));
      }
    }

    auto mean_rep = mean.view({batch, channels, 1}).repeat(seq_len, 2);
    auto std_rep = std.view({batch, channels, 1}).repeat(seq_len, 2);

    auto attention = nn::functional::concat({hidden_states, mean_rep, std_rep}, 1);
    attention = tdnn_(attention)[0];
    attention = tanh_(attention);
    attention = conv_(attention);
    attention = nn::functional::softmax(attention, 2);

    auto out_mean = Tensor::empty({batch, channels}, kFloat32, kCPU).alloc();
    auto out_std = Tensor::empty({batch, channels}, kFloat32, kCPU).alloc();
    auto* out_mean_ptr = out_mean.ptr<float>();
    auto* out_std_ptr = out_std.ptr<float>();
    const auto* attn_ptr = attention.ptr<float>();

    for (int32_t b = 0; b < batch; ++b) {
      for (int32_t c = 0; c < channels; ++c) {
        const int32_t base = b * stride_b + c * stride_c;
        float m = 0.0f;
        for (int32_t t = 0; t < seq_len; ++t) { m += attn_ptr[base + t] * x_ptr[base + t]; }
        out_mean_ptr[b * channels + c] = m;

        float var = 0.0f;
        for (int32_t t = 0; t < seq_len; ++t) {
          float diff = x_ptr[base + t] - m;
          var += attn_ptr[base + t] * diff * diff;
        }
        out_std_ptr[b * channels + c] = std::sqrt(std::max(var, 1e-12f));
      }
    }

    auto pooled = nn::functional::concat({out_mean, out_std}, 1).view({batch, channels * 2, 1});
    return {pooled};
  }

 private:
  int32_t channels_ = 0;
  TimeDelayNetBlock tdnn_;
  nn::Tanh tanh_;
  nn::Conv1D conv_;
};

class SqueezeExcitationRes2NetBlock final : public nn::Module {
 public:
  SqueezeExcitationRes2NetBlock() = default;
  SqueezeExcitationRes2NetBlock(const std::string& name, int32_t in_channels, int32_t out_channels, int32_t res2net_scale,
                                int32_t se_channels, int32_t kernel_size, int32_t dilation)
      : nn::Module(name), out_channels_(out_channels) {
    tdnn1_ = reg<TimeDelayNetBlock>("tdnn1", in_channels, out_channels, 1, 1);
    res2net_block_ = reg<Res2NetBlock>("res2net_block", out_channels, out_channels, res2net_scale, kernel_size, dilation);
    tdnn2_ = reg<TimeDelayNetBlock>("tdnn2", out_channels, out_channels, 1, 1);
    se_block_ = reg<SqueezeExcitationBlock>("se_block", out_channels, se_channels, out_channels);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto hidden_state = inputs[0];
    auto residual = hidden_state;

    hidden_state = tdnn1_(hidden_state)[0];
    hidden_state = res2net_block_(hidden_state)[0];
    hidden_state = tdnn2_(hidden_state)[0];
    hidden_state = se_block_(hidden_state)[0];
    hidden_state = hidden_state + residual;
    return {hidden_state};
  }

 private:
  int32_t out_channels_ = 0;
  TimeDelayNetBlock tdnn1_;
  Res2NetBlock res2net_block_;
  TimeDelayNetBlock tdnn2_;
  SqueezeExcitationBlock se_block_;
};

class ECAPA_TimeDelayNet final : public nn::Module {
 public:
  ECAPA_TimeDelayNet() = default;
  explicit ECAPA_TimeDelayNet(const std::string& name, const Qwen2_5OmniDiTConfig& cfg) : nn::Module(name) {
    if (cfg.enc_channels.size() != cfg.enc_kernel_sizes.size() || cfg.enc_channels.size() != cfg.enc_dilations.size()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "enc_channels, enc_kernel_sizes and enc_dilations should have same length");
    }

    if (cfg.enc_channels.empty()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "enc_channels should not be empty");
    }

    const int32_t num_blocks = static_cast<int32_t>(cfg.enc_channels.size());
    tdnn0_ = reg<TimeDelayNetBlock>("blocks.0", cfg.mel_dim, cfg.enc_channels[0], cfg.enc_kernel_sizes[0], cfg.enc_dilations[0]);

    for (int32_t i = 1; i < num_blocks - 1; ++i) {
      se_blocks_.emplace_back(reg<SqueezeExcitationRes2NetBlock>(
          "blocks." + std::to_string(i),
          cfg.enc_channels[i - 1],
          cfg.enc_channels[i],
          cfg.enc_res2net_scale,
          cfg.enc_se_channels,
          cfg.enc_kernel_sizes[i],
          cfg.enc_dilations[i]));
    }

    mfa_ = reg<TimeDelayNetBlock>("mfa", cfg.enc_channels.back(), cfg.enc_channels.back(), cfg.enc_kernel_sizes.back(),
                                 cfg.enc_dilations.back());
    asp_ = reg<AttentiveStatisticsPooling>("asp", cfg.enc_channels.back(), cfg.enc_attention_channels);
    fc_ = reg<nn::Conv1D>("fc", cfg.enc_channels.back() * 2, cfg.enc_dim, 1, 1, 0, 1, 1, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto hidden_states = inputs[0];
    MLLM_RT_ASSERT_EQ(hidden_states.dtype(), kFloat32);
    MLLM_RT_ASSERT_EQ(hidden_states.device(), kCPU);

    hidden_states = hidden_states.transpose(1, 2);

    std::vector<Tensor> hidden_states_list;
    hidden_states = tdnn0_(hidden_states)[0];
    hidden_states_list.push_back(hidden_states);

    for (auto& block : se_blocks_) {
      hidden_states = block(hidden_states)[0];
      hidden_states_list.push_back(hidden_states);
    }

    if (hidden_states_list.size() <= 1) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "ECAPA_TimeDelayNet expects at least 2 blocks.");
    }

    std::vector<Tensor> mfa_inputs;
    for (size_t i = 1; i < hidden_states_list.size(); ++i) { mfa_inputs.push_back(hidden_states_list[i]); }
    hidden_states = nn::functional::concat(mfa_inputs, 1);
    hidden_states = mfa_(hidden_states)[0];
    hidden_states = asp_(hidden_states)[0];
    hidden_states = fc_(hidden_states);
    hidden_states = hidden_states.squeeze(-1);

    return {hidden_states};
  }

 private:
  TimeDelayNetBlock tdnn0_;
  std::vector<SqueezeExcitationRes2NetBlock> se_blocks_;
  TimeDelayNetBlock mfa_;
  AttentiveStatisticsPooling asp_;
  nn::Conv1D fc_;
};

class DiTInputEmbedding final : public nn::Module {
 public:
  DiTInputEmbedding() = default;
  explicit DiTInputEmbedding(const std::string& name, const Qwen2_5OmniDiTConfig& cfg) : nn::Module(name) {
    const int32_t in_dim = cfg.mel_dim + cfg.enc_dim + cfg.enc_emb_dim + cfg.emb_dim;
    proj_ = reg<nn::Linear>("proj", in_dim, cfg.hidden_size, true);
    spk_encoder_ = reg<ECAPA_TimeDelayNet>("spk_encoder", cfg);
  }

  Tensor forward(const Tensor& hidden_states, const Tensor& speaker_embedding, const Tensor& condition_vector, const Tensor& code_embed,
                 bool drop_audio_cond, const Tensor& code_embed_uncond, bool apply_cfg) {
    auto x = hidden_states;
    auto spk = speaker_embedding;
    auto cond = condition_vector;
    auto code = code_embed;

    if (apply_cfg) {
      x = nn::functional::concat({x, x}, 0);
      spk = nn::functional::concat({spk, Tensor::zeros(spk.shape(), spk.dtype(), spk.device())}, 0);
      cond = nn::functional::concat({cond, Tensor::zeros(cond.shape(), cond.dtype(), cond.device())}, 0);
      code = nn::functional::concat({code, code_embed_uncond}, 0);
    } else if (drop_audio_cond) {
      cond = Tensor::zeros(cond.shape(), cond.dtype(), cond.device());
      spk = Tensor::zeros(spk.shape(), spk.dtype(), spk.device());
    }

    auto cond_embed = spk_encoder_(cond)[0];
    const int32_t seq_len = x.shape()[1];
    cond_embed = cond_embed.view({cond_embed.shape()[0], 1, cond_embed.shape()[1]}).repeat(seq_len, 1);

    auto merged = nn::functional::concat({x, cond_embed, code, spk}, -1);
    auto out = proj_(merged);
    return out;
  }

 private:
  nn::Linear proj_;
  ECAPA_TimeDelayNet spk_encoder_;
};

class DiTCodecEmbedding final : public nn::Module {
 public:
  DiTCodecEmbedding() = default;
  DiTCodecEmbedding(const std::string& name, int32_t codec_num_embeds, int32_t codec_dim, int32_t repeats)
      : nn::Module(name), repeats_(repeats) {
    codec_embed_ = reg<nn::Embedding>("codec_embed", codec_num_embeds + 1, codec_dim);
  }

  Tensor forward(const Tensor& code, bool drop_code) {
    Tensor code_ids = code;
    if (drop_code) { code_ids = Tensor::zeros(code.shape(), code.dtype(), code.device()); }
    auto code_embed = codec_embed_(code_ids);
    return repeatInterleave(code_embed, repeats_, 1);
  }

 private:
  int32_t repeats_ = 1;
  nn::Embedding codec_embed_;
};

class Qwen2_5_OmniAdaLayerNormZero final : public nn::Module {
 public:
  Qwen2_5_OmniAdaLayerNormZero() = default;
  Qwen2_5_OmniAdaLayerNormZero(const std::string& name, int32_t dim) : nn::Module(name) {
    silu_ = reg<nn::SiLU>("silu");
    linear_ = reg<nn::Linear>("linear", dim, dim * 6, true);
    norm_ = reg<nn::LayerNorm>("norm", std::vector<int32_t>{dim}, false, false, 1e-6f);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto hidden_states = inputs[0];
    auto emb = inputs[1];
    emb = linear_(silu_(emb));

    auto chunks = nn::functional::chunk<6>(emb, 1);
    auto shift_msa = chunks[0];
    auto scale_msa = chunks[1];
    auto gate_msa = chunks[2];
    auto shift_mlp = chunks[3];
    auto scale_mlp = chunks[4];
    auto gate_mlp = chunks[5];

    auto normed = norm_(hidden_states);
    const int32_t seq_len = hidden_states.shape()[1];
    auto scale = scale_msa.view({scale_msa.shape()[0], 1, scale_msa.shape()[1]}).repeat(seq_len, 1);
    auto shift = shift_msa.view({shift_msa.shape()[0], 1, shift_msa.shape()[1]}).repeat(seq_len, 1);
    normed = normed * (scale + 1.0f) + shift;

    return {normed, gate_msa, shift_mlp, scale_mlp, gate_mlp};
  }

 private:
  nn::SiLU silu_;
  nn::Linear linear_;
  nn::LayerNorm norm_;
};

class Qwen2_5_OmniAdaLayerNormZero_Final final : public nn::Module {
 public:
  Qwen2_5_OmniAdaLayerNormZero_Final() = default;
  Qwen2_5_OmniAdaLayerNormZero_Final(const std::string& name, int32_t dim) : nn::Module(name) {
    silu_ = reg<nn::SiLU>("silu");
    linear_ = reg<nn::Linear>("linear", dim, dim * 2, true);
    norm_ = reg<nn::LayerNorm>("norm", std::vector<int32_t>{dim}, false, false, 1e-6f);
  }

  Tensor forward(const Tensor& hidden_states, const Tensor& emb) {
    auto emb_out = linear_(silu_(emb));
    auto chunks = nn::functional::chunk<2>(emb_out, 1);
    auto scale = chunks[0];
    auto shift = chunks[1];

    auto normed = norm_(hidden_states);
    const int32_t seq_len = hidden_states.shape()[1];
    scale = scale.view({scale.shape()[0], 1, scale.shape()[1]}).repeat(seq_len, 1);
    shift = shift.view({shift.shape()[0], 1, shift.shape()[1]}).repeat(seq_len, 1);
    normed = normed * (scale + 1.0f) + shift;
    return normed;
  }

 private:
  nn::SiLU silu_;
  nn::Linear linear_;
  nn::LayerNorm norm_;
};

class DiTMLP final : public nn::Module {
 public:
  DiTMLP() = default;
  DiTMLP(const std::string& name, int32_t dim, int32_t mult) : nn::Module(name) {
    const int32_t inner_dim = dim * mult;
    fc1_ = reg<nn::Linear>("ff.0", dim, inner_dim, true);
    act_ = reg<nn::GELU>("ff.1");
    fc2_ = reg<nn::Linear>("ff.3", inner_dim, dim, true);
  }

  Tensor forward(const Tensor& hidden_states) {
    auto x = fc1_(hidden_states);
    x = act_(x);
    x = fc2_(x);
    return x;
  }

 private:
  nn::Linear fc1_;
  nn::GELU act_;
  nn::Linear fc2_;
};

inline void applyRotaryPosEmbFirstHead(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin) {
  MLLM_RT_ASSERT_EQ(q.device(), kCPU);
  MLLM_RT_ASSERT_EQ(k.device(), kCPU);
  MLLM_RT_ASSERT_EQ(cos.device(), kCPU);
  MLLM_RT_ASSERT_EQ(sin.device(), kCPU);
  MLLM_RT_ASSERT_EQ(q.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(k.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(cos.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(sin.dtype(), kFloat32);

  const int32_t batch = q.shape()[0];
  const int32_t heads = q.shape()[1];
  const int32_t seq_len = q.shape()[2];
  const int32_t head_dim = q.shape()[3];
  MLLM_RT_ASSERT_EQ(head_dim % 2, 0);
  MLLM_RT_ASSERT_EQ(cos.shape()[0], batch);
  MLLM_RT_ASSERT_EQ(cos.shape()[1], seq_len);
  MLLM_RT_ASSERT_EQ(cos.shape()[2], head_dim);

  const auto* cos_ptr = cos.ptr<float>();
  const auto* sin_ptr = sin.ptr<float>();
  auto* q_ptr = q.ptr<float>();
  auto* k_ptr = k.ptr<float>();

  const int64_t stride_q_b = static_cast<int64_t>(heads) * seq_len * head_dim;
  const int64_t stride_q_h = static_cast<int64_t>(seq_len) * head_dim;
  const int64_t stride_q_s = head_dim;

  const int64_t stride_cos_b = static_cast<int64_t>(seq_len) * head_dim;
  const int64_t stride_cos_s = head_dim;

  for (int32_t b = 0; b < batch; ++b) {
    const int64_t q_base_b = static_cast<int64_t>(b) * stride_q_b;
    const int64_t cos_base_b = static_cast<int64_t>(b) * stride_cos_b;
    for (int32_t s = 0; s < seq_len; ++s) {
      float* q_row = q_ptr + q_base_b + 0 * stride_q_h + static_cast<int64_t>(s) * stride_q_s;
      float* k_row = k_ptr + q_base_b + 0 * stride_q_h + static_cast<int64_t>(s) * stride_q_s;
      const float* cos_row = cos_ptr + cos_base_b + static_cast<int64_t>(s) * stride_cos_s;
      const float* sin_row = sin_ptr + cos_base_b + static_cast<int64_t>(s) * stride_cos_s;
      for (int32_t d = 0; d < head_dim; d += 2) {
        const float c = cos_row[d];
        const float ss = sin_row[d];
        const float q1 = q_row[d];
        const float q2 = q_row[d + 1];
        const float k1 = k_row[d];
        const float k2 = k_row[d + 1];
        q_row[d] = q1 * c - q2 * ss;
        q_row[d + 1] = q1 * ss + q2 * c;
        k_row[d] = k1 * c - k2 * ss;
        k_row[d + 1] = k1 * ss + k2 * c;
      }
    }
  }
}

inline Tensor makeBlockDiff(int32_t batch, int32_t heads, int32_t seq_len, int32_t block_size) {
  (void)heads;
  MLLM_RT_ASSERT(block_size > 0);
  std::vector<int32_t> block_indices(seq_len, 0);
  for (int32_t i = 0; i < seq_len; ++i) { block_indices[i] = i / block_size; }

  std::vector<float> base(static_cast<size_t>(seq_len) * seq_len, 0.0f);
  for (int32_t i = 0; i < seq_len; ++i) {
    for (int32_t j = 0; j < seq_len; ++j) {
      base[static_cast<size_t>(i) * seq_len + j] = static_cast<float>(block_indices[j] - block_indices[i]);
    }
  }

  // Use a broadcast-friendly shape to avoid materializing head copies while keeping naive broadcast support.
  auto out = Tensor::empty({batch, 1, seq_len, seq_len}, kFloat32, kCPU).alloc();
  const int64_t block_stride = static_cast<int64_t>(seq_len) * seq_len;
  auto* out_ptr = out.ptr<float>();
  for (int32_t b = 0; b < batch; ++b) {
    float* dst = out_ptr + static_cast<int64_t>(b) * block_stride;
    std::memcpy(dst, base.data(), sizeof(float) * base.size());
  }
  return out;
}

inline Tensor makeBlockMask(const Tensor& block_diff, int32_t look_backward_block, int32_t look_ahead_block) {
  MLLM_RT_ASSERT_EQ(block_diff.device(), kCPU);
  MLLM_RT_ASSERT_EQ(block_diff.dtype(), kFloat32);

  auto mask = Tensor::empty(block_diff.shape(), kFloat32, kCPU).alloc();
  const auto* src = block_diff.ptr<float>();
  auto* dst = mask.ptr<float>();
  const int64_t numel = block_diff.numel();
  const float lower = -static_cast<float>(look_backward_block);
  const float upper = static_cast<float>(look_ahead_block);

  MLLM_CONDITIONAL_PARALLEL_FOR(numel > 1024, 4, idx, 0, numel, 1, {
    const float v = src[idx];
    dst[idx] = (v >= lower && v <= upper) ? 0.0f : -1e4f;
  });
  return mask;
}

class DiTAttention final : public nn::Module {
 public:
  DiTAttention() = default;
  explicit DiTAttention(const std::string& name, const Qwen2_5OmniDiTConfig& cfg) : nn::Module(name), cfg_(cfg) {
    dim_ = cfg.hidden_size;
    heads_ = cfg.num_attention_heads;
    head_dim_ = cfg.head_dim;
    inner_dim_ = head_dim_ * heads_;

    to_q_ = reg<nn::Linear>("to_q", dim_, inner_dim_, true);
    to_k_ = reg<nn::Linear>("to_k", dim_, inner_dim_, true);
    to_v_ = reg<nn::Linear>("to_v", dim_, inner_dim_, true);
    to_out_ = reg<nn::Linear>("to_out.0", inner_dim_, dim_, true);
  }

  Tensor forward(const Tensor& hidden_states, const std::pair<Tensor, Tensor>& position_embeddings, const Tensor& attention_mask) {
    auto query = to_q_(hidden_states);
    auto key = to_k_(hidden_states);
    auto value = to_v_(hidden_states);

    const int32_t batch = hidden_states.shape()[0];
    const int32_t seq_len = hidden_states.shape()[1];

    query = query.view({batch, seq_len, heads_, head_dim_}).transpose(1, 2);
    key = key.view({batch, seq_len, heads_, head_dim_}).transpose(1, 2);
    value = value.view({batch, seq_len, heads_, head_dim_}).transpose(1, 2);

    if (!position_embeddings.first.isNil()) {
      applyRotaryPosEmbFirstHead(query, key, position_embeddings.first, position_embeddings.second);
    }

    auto attn_output = nn::functional::scaledDotProductAttention(query, key, value, attention_mask);
    attn_output = attn_output.transpose(1, 2).view({batch, seq_len, inner_dim_});
    attn_output = to_out_(attn_output);
    return attn_output;
  }

 private:
  Qwen2_5OmniDiTConfig cfg_;
  int32_t dim_ = 0;
  int32_t heads_ = 0;
  int32_t head_dim_ = 0;
  int32_t inner_dim_ = 0;
  nn::Linear to_q_;
  nn::Linear to_k_;
  nn::Linear to_v_;
  nn::Linear to_out_;
};

class SinusPositionEmbedding final : public nn::Module {
 public:
  SinusPositionEmbedding() = default;
  explicit SinusPositionEmbedding(const std::string& name, int32_t dim) : nn::Module(name), dim_(dim) {}

  Tensor forward(const Tensor& hidden_states, float scale = 1000.0f) {
    MLLM_RT_ASSERT_EQ(hidden_states.device(), kCPU);
    MLLM_RT_ASSERT_EQ(hidden_states.dtype(), kFloat32);

    const int32_t batch = hidden_states.shape()[0];
    const int32_t half_dim = dim_ / 2;
    auto out = Tensor::empty({batch, dim_}, kFloat32, kCPU).alloc();
    auto* out_ptr = out.ptr<float>();
    const auto* hs_ptr = hidden_states.ptr<float>();

    const float emb = std::log(10000.0f) / static_cast<float>(half_dim - 1);
    std::vector<float> freqs(half_dim);
    for (int32_t i = 0; i < half_dim; ++i) { freqs[i] = std::exp(-emb * static_cast<float>(i)); }

    for (int32_t b = 0; b < batch; ++b) {
      const float t = hs_ptr[b] * scale;
      float* row = out_ptr + static_cast<int64_t>(b) * dim_;
      for (int32_t i = 0; i < half_dim; ++i) {
        const float val = t * freqs[i];
        row[i] = std::sin(val);
        row[i + half_dim] = std::cos(val);
      }
    }

    return out;
  }

 private:
  int32_t dim_ = 0;
};

class DiTTimestepEmbedding final : public nn::Module {
 public:
  DiTTimestepEmbedding() = default;
  explicit DiTTimestepEmbedding(const std::string& name, int32_t dim, int32_t freq_embed_dim = 256)
      : nn::Module(name), freq_embed_dim_(freq_embed_dim) {
    time_embed_ = reg<SinusPositionEmbedding>("time_embed", freq_embed_dim_);
    fc1_ = reg<nn::Linear>("time_mlp.0", freq_embed_dim_, dim, true);
    act_ = reg<nn::SiLU>("time_mlp.1");
    fc2_ = reg<nn::Linear>("time_mlp.2", dim, dim, true);
  }

  Tensor forward(const Tensor& timestep) {
    auto time_hidden = time_embed_.forward(timestep);
    time_hidden = fc1_(time_hidden);
    time_hidden = act_(time_hidden);
    time_hidden = fc2_(time_hidden);
    return time_hidden;
  }

 private:
  int32_t freq_embed_dim_ = 256;
  SinusPositionEmbedding time_embed_;
  nn::Linear fc1_;
  nn::SiLU act_;
  nn::Linear fc2_;
};

class DiTDecoderLayer final : public nn::Module {
 public:
  DiTDecoderLayer() = default;
  DiTDecoderLayer(const std::string& name, const Qwen2_5OmniDiTConfig& cfg, int32_t look_ahead_block, int32_t look_backward_block)
      : nn::Module(name), look_ahead_block_(look_ahead_block), look_backward_block_(look_backward_block) {
    attn_norm_ = reg<Qwen2_5_OmniAdaLayerNormZero>("attn_norm", cfg.hidden_size);
    attn_ = reg<DiTAttention>("attn", cfg);
    ff_norm_ = reg<nn::LayerNorm>("ff_norm", std::vector<int32_t>{cfg.hidden_size}, false, false, 1e-6f);
    ff_ = reg<DiTMLP>("ff", cfg.hidden_size, cfg.ff_mult);
  }

  Tensor forward(const Tensor& hidden_states, const Tensor& timestep, const std::pair<Tensor, Tensor>& position_embeddings,
                 const Tensor& block_diff) {
    auto attn_norm_out = attn_norm_(hidden_states, timestep);
    auto norm = attn_norm_out[0];
    auto gate_msa = attn_norm_out[1];
    auto shift_mlp = attn_norm_out[2];
    auto scale_mlp = attn_norm_out[3];
    auto gate_mlp = attn_norm_out[4];

    Tensor attn_mask = Tensor::nil();
    if (!block_diff.isNil()) { attn_mask = makeBlockMask(block_diff, look_backward_block_, look_ahead_block_); }
    auto attn_output = attn_.forward(norm, position_embeddings, attn_mask);

    auto gate_msa_rep = gate_msa.view({gate_msa.shape()[0], 1, gate_msa.shape()[1]}).repeat(hidden_states.shape()[1], 1);
    auto x = Tensor(hidden_states);
    x = x + gate_msa_rep * attn_output;

    auto norm_ff = ff_norm_(x);
    auto scale_rep = scale_mlp.view({scale_mlp.shape()[0], 1, scale_mlp.shape()[1]}).repeat(x.shape()[1], 1);
    auto shift_rep = shift_mlp.view({shift_mlp.shape()[0], 1, shift_mlp.shape()[1]}).repeat(x.shape()[1], 1);
    norm_ff = norm_ff * (scale_rep + 1.0f) + shift_rep;
    auto ff_output = ff_.forward(norm_ff);
    auto gate_mlp_rep = gate_mlp.view({gate_mlp.shape()[0], 1, gate_mlp.shape()[1]}).repeat(x.shape()[1], 1);
    x = x + gate_mlp_rep * ff_output;
    return x;
  }

 private:
  Qwen2_5_OmniAdaLayerNormZero attn_norm_;
  DiTAttention attn_;
  nn::LayerNorm ff_norm_;
  DiTMLP ff_;
  int32_t look_ahead_block_ = 0;
  int32_t look_backward_block_ = 0;
};

class Qwen2_5OmniDiTRotaryEmbedding final : public nn::Module {
 public:
  Qwen2_5OmniDiTRotaryEmbedding() = default;
  explicit Qwen2_5OmniDiTRotaryEmbedding(const std::string& name, const Qwen2_5OmniDiTConfig& cfg) : nn::Module(name), cfg_(cfg) {
    const int32_t dim = cfg.head_dim;
    inv_freq_ = reg<nn::Param>("inv_freq", getModuleName() + ".inv_freq", std::vector<int32_t>{dim / 2});
    attention_scaling_ = 1.0f;

    auto inv = inv_freq_.weight();
    if (!inv.isNil() && inv.numel() == 0) {
      inv = Tensor::empty({dim / 2}, kFloat32, kCPU).alloc();
      inv_freq_.weight().copy2(inv);
    }
  }

  std::pair<Tensor, Tensor> forward(const Tensor& x, const Tensor& position_ids) {
    MLLM_RT_ASSERT_EQ(x.device(), kCPU);
    MLLM_RT_ASSERT_EQ(position_ids.device(), kCPU);
    MLLM_RT_ASSERT_EQ(position_ids.dtype(), kInt64);

    const int32_t batch = position_ids.shape()[0];
    const int32_t seq_len = position_ids.shape()[1];
    auto inv_freq = inv_freq_.weight();
    if (inv_freq.isNil() || inv_freq.numel() == 0) {
      const int32_t dim = cfg_.head_dim;
      inv_freq = Tensor::empty({dim / 2}, kFloat32, kCPU).alloc();
      auto* ptr = inv_freq.ptr<float>();
      for (int32_t i = 0; i < dim / 2; ++i) {
        ptr[i] = 1.0f / std::pow(cfg_.rope_theta, 2.0f * i / static_cast<float>(dim));
      }
    }

    const int32_t half_dim = inv_freq.shape()[0];
    auto cos = Tensor::empty({batch, seq_len, half_dim * 2}, kFloat32, kCPU).alloc();
    auto sin = Tensor::empty({batch, seq_len, half_dim * 2}, kFloat32, kCPU).alloc();

    const auto* inv_ptr = inv_freq.ptr<float>();
    const auto* pos_ptr = position_ids.ptr<int64_t>();
    auto* cos_ptr = cos.ptr<float>();
    auto* sin_ptr = sin.ptr<float>();

    const int64_t stride_pos_b = seq_len;
    const int64_t stride_cos_b = static_cast<int64_t>(seq_len) * half_dim * 2;
    const int64_t stride_cos_s = half_dim * 2;

    for (int32_t b = 0; b < batch; ++b) {
      const int64_t pos_base = static_cast<int64_t>(b) * stride_pos_b;
      const int64_t out_base = static_cast<int64_t>(b) * stride_cos_b;
      for (int32_t s = 0; s < seq_len; ++s) {
        const float position = static_cast<float>(pos_ptr[pos_base + s]);
        float* cos_row = cos_ptr + out_base + static_cast<int64_t>(s) * stride_cos_s;
        float* sin_row = sin_ptr + out_base + static_cast<int64_t>(s) * stride_cos_s;
        for (int32_t d = 0; d < half_dim; ++d) {
          const float freq = inv_ptr[d] * position;
          const float c = std::cos(freq) * attention_scaling_;
          const float ss = std::sin(freq) * attention_scaling_;
          cos_row[d] = c;
          cos_row[d + half_dim] = c;
          sin_row[d] = ss;
          sin_row[d + half_dim] = ss;
        }
      }
    }

    return {cos, sin};
  }

 private:
  Qwen2_5OmniDiTConfig cfg_;
  nn::Param inv_freq_;
  float attention_scaling_ = 1.0f;
};

class RungeKutta4ODESolver {
 public:
  using Function = std::function<Tensor(float, const Tensor&)>;

  RungeKutta4ODESolver(Function function, Tensor initial_value)
      : function_(std::move(function)), initial_value_(std::move(initial_value)) {}

  Tensor integrate(const std::vector<float>& time_points) {
    auto current_value = initial_value_;
    if (time_points.size() < 2) { return current_value; }

    for (size_t i = 0; i + 1 < time_points.size(); ++i) {
      const float time_start = time_points[i];
      const float time_end = time_points[i + 1];
      const float time_step = time_end - time_start;

      auto k1 = function_(time_start, current_value);
      auto k2 = function_(time_start + time_step * one_third_, current_value + k1 * (time_step * one_third_));
      auto k3 = function_(time_start + time_step * two_thirds_,
                          current_value + (k2 - k1 * one_third_) * time_step);
      auto k4 = function_(time_end, current_value + (k1 - k2 + k3) * time_step);

      auto delta = (k1 + (k2 + k3) * 3.0f + k4) * (time_step / 8.0f);
      current_value = current_value + delta;
    }

    return current_value;
  }

 private:
  Function function_;
  Tensor initial_value_;
  float one_third_ = 1.0f / 3.0f;
  float two_thirds_ = 2.0f / 3.0f;
};

class Qwen2_5OmniToken2WavDiTModel final : public nn::Module {
 public:
  Qwen2_5OmniToken2WavDiTModel() = default;
  explicit Qwen2_5OmniToken2WavDiTModel(const std::string& name, const Qwen2_5OmniDiTConfig& cfg) : nn::Module(name), cfg_(cfg) {
    mel_dim_ = cfg.mel_dim;
    repeats_ = cfg.repeats;
    block_size_ = cfg.block_size;
    num_attention_heads_ = cfg.num_attention_heads;

    time_embed_ = reg<DiTTimestepEmbedding>("time_embed", cfg.hidden_size);
    text_embed_ = reg<DiTCodecEmbedding>("text_embed", cfg.num_embeds, cfg.emb_dim, cfg.repeats);
    input_embed_ = reg<DiTInputEmbedding>("input_embed", cfg);
    rotary_embed_ = reg<Qwen2_5OmniDiTRotaryEmbedding>("rotary_embed", cfg);

    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
      const bool look_ahead = std::find(cfg.look_ahead_layers.begin(), cfg.look_ahead_layers.end(), i) != cfg.look_ahead_layers.end();
      const bool look_backward =
          std::find(cfg.look_backward_layers.begin(), cfg.look_backward_layers.end(), i) != cfg.look_backward_layers.end();
      transformer_blocks_.emplace_back(reg<DiTDecoderLayer>("transformer_blocks." + std::to_string(i), cfg, look_ahead ? 1 : 0,
                                                           look_backward ? 1 : 0));
    }

    norm_out_ = reg<Qwen2_5_OmniAdaLayerNormZero_Final>("norm_out", cfg.hidden_size);
    proj_out_ = reg<nn::Linear>("proj_out", cfg.hidden_size, cfg.mel_dim, true);
  }

  Tensor forward(const Tensor& hidden_states, const Tensor& condition_vector, const Tensor& speaker_embedding, const Tensor& quantized_code,
                 const Tensor& time_step, bool drop_audio_conditioning, bool drop_code, bool apply_cfg) {
    Tensor timestep = time_step;
    if (timestep.shape().empty()) { timestep = timestep.view({1}); }
    if (timestep.shape().size() == 1 && timestep.shape()[0] == 1 && hidden_states.shape()[0] > 1) {
      timestep = timestep.repeat(hidden_states.shape()[0], 0);
    }

    auto time_embedding = time_embed_.forward(timestep);
    auto text_embedding = text_embed_.forward(quantized_code, apply_cfg ? false : drop_code);
    Tensor text_embedding_uncond = Tensor::nil();
    if (apply_cfg) { text_embedding_uncond = text_embed_.forward(quantized_code, true); }

    auto x = input_embed_.forward(hidden_states, speaker_embedding, condition_vector, text_embedding, drop_audio_conditioning,
                                  text_embedding_uncond, apply_cfg);

    const int32_t seq_len = x.shape()[1];
    auto position_ids = Tensor::empty({x.shape()[0], seq_len}, kInt64, kCPU).alloc();
    auto* pos_ptr = position_ids.ptr<int64_t>();
    for (int32_t b = 0; b < position_ids.shape()[0]; ++b) {
      for (int32_t s = 0; s < seq_len; ++s) { pos_ptr[b * seq_len + s] = s; }
    }

    auto position_embeddings = rotary_embed_.forward(x, position_ids);
    auto block_diff = makeBlockDiff(x.shape()[0], num_attention_heads_, seq_len, block_size_);

    for (auto& block : transformer_blocks_) { x = block.forward(x, time_embedding, position_embeddings, block_diff); }

    x = norm_out_.forward(x, time_embedding);
    x = proj_out_(x);
    return x;
  }

  Tensor sample(const Tensor& conditioning_vector, const Tensor& reference_mel, const Tensor& quantized_code, int32_t num_steps,
                float guidance_scale, float sway_coefficient) {
    const int32_t max_duration = quantized_code.shape()[1] * repeats_;
    auto initial_state = randomNormal({1, max_duration, mel_dim_});

    const int32_t batch = reference_mel.shape()[0];
    if (batch != 1) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Only batch size = 1 is supported for Qwen2.5-Omni token2wav."); }

    auto cond = Tensor(conditioning_vector);
    cond = cond.view({batch, 1, conditioning_vector.shape()[1]}).repeat(max_duration, 1);

    auto ode_function = [&](float time_step, const Tensor& hidden) -> Tensor {
      auto t = Tensor::empty({1}, kFloat32, kCPU).alloc();
      t.ptr<float>()[0] = time_step;

      if (guidance_scale < 1e-5f) {
        return forward(hidden, reference_mel, cond, quantized_code, t, false, false, false);
      }

      auto model_output = forward(hidden, reference_mel, cond, quantized_code, t, false, false, true);
      auto outputs = nn::functional::chunk<2>(model_output, 0);
      return outputs[0] + (outputs[0] - outputs[1]) * guidance_scale;
    };

    auto time_points_tensor = linspace(0.0f, 1.0f, num_steps);
    std::vector<float> time_points(static_cast<size_t>(num_steps));
    const auto* tp_ptr = time_points_tensor.ptr<float>();
    for (int32_t i = 0; i < num_steps; ++i) { time_points[i] = tp_ptr[i]; }

    if (sway_coefficient != 0.0f) {
      for (auto& t : time_points) {
        t = t + sway_coefficient * (std::cos(kPi / 2.0f * t) - 1.0f + t);
      }
    }

    RungeKutta4ODESolver solver(ode_function, initial_state);
    auto generated = solver.integrate(time_points);
    auto mel = generated.permute({0, 2, 1});
    if (!mel.isContiguous()) { mel = mel.contiguous(); }
    return mel;
  }

 private:
  Qwen2_5OmniDiTConfig cfg_;
  int32_t mel_dim_ = 0;
  int32_t repeats_ = 1;
  int32_t block_size_ = 1;
  int32_t num_attention_heads_ = 1;

  DiTTimestepEmbedding time_embed_;
  DiTCodecEmbedding text_embed_;
  DiTInputEmbedding input_embed_;
  Qwen2_5OmniDiTRotaryEmbedding rotary_embed_;
  std::vector<DiTDecoderLayer> transformer_blocks_;
  Qwen2_5_OmniAdaLayerNormZero_Final norm_out_;
  nn::Linear proj_out_;
};

class AMPBlock final : public nn::Module {
 public:
  AMPBlock() = default;
  AMPBlock(const std::string& name, int32_t channels, int32_t kernel_size, const std::vector<int32_t>& dilations)
      : nn::Module(name) {
    if (dilations.size() != 3) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "AMPBlock expects 3 dilation values."); }

    convs1_.emplace_back(reg<nn::Conv1D>("convs1.0", channels, channels, kernel_size, 1, getPadding(kernel_size, dilations[0]),
                                         dilations[0], 1, true));
    convs1_.emplace_back(reg<nn::Conv1D>("convs1.1", channels, channels, kernel_size, 1, getPadding(kernel_size, dilations[1]),
                                         dilations[1], 1, true));
    convs1_.emplace_back(reg<nn::Conv1D>("convs1.2", channels, channels, kernel_size, 1, getPadding(kernel_size, dilations[2]),
                                         dilations[2], 1, true));

    convs2_.emplace_back(reg<nn::Conv1D>("convs2.0", channels, channels, kernel_size, 1, getPadding(kernel_size, 1), 1, 1, true));
    convs2_.emplace_back(reg<nn::Conv1D>("convs2.1", channels, channels, kernel_size, 1, getPadding(kernel_size, 1), 1, 1, true));
    convs2_.emplace_back(reg<nn::Conv1D>("convs2.2", channels, channels, kernel_size, 1, getPadding(kernel_size, 1), 1, 1, true));

    const int32_t num_layers = static_cast<int32_t>(convs1_.size() + convs2_.size());
    for (int32_t i = 0; i < num_layers; ++i) {
      activations_.emplace_back(reg<TorchActivation1d>("activations." + std::to_string(i), channels));
    }
  }

  Tensor forward(const Tensor& hidden_states) {
    auto out = hidden_states;
    const int32_t num_blocks = static_cast<int32_t>(convs1_.size());
    for (int32_t i = 0; i < num_blocks; ++i) {
      auto residual = out;
      auto x = activations_[i * 2].forward({out}, {})[0];
      x = convs1_[i](x);
      x = activations_[i * 2 + 1].forward({x}, {})[0];
      x = convs2_[i](x);
      out = residual + x;
    }
    return out;
  }

 private:
  static int32_t getPadding(int32_t kernel_size, int32_t dilation) {
    return static_cast<int32_t>((kernel_size * dilation - dilation) / 2);
  }

  std::vector<nn::Conv1D> convs1_;
  std::vector<nn::Conv1D> convs2_;
  std::vector<TorchActivation1d> activations_;
};

class Qwen2_5OmniToken2WavBigVGANModel final : public nn::Module {
 public:
  Qwen2_5OmniToken2WavBigVGANModel() = default;
  explicit Qwen2_5OmniToken2WavBigVGANModel(const std::string& name, const Qwen2_5OmniBigVGANConfig& cfg) : nn::Module(name), cfg_(cfg) {
    num_residual_blocks_ = static_cast<int32_t>(cfg.resblock_kernel_sizes.size());
    num_upsample_layers_ = static_cast<int32_t>(cfg.upsample_rates.size());

    conv_pre_ = reg<nn::Conv1D>("conv_pre", cfg.mel_dim, cfg.upsample_initial_channel, 7, 1, 3, 1, 1, true);

    for (int32_t layer_idx = 0; layer_idx < num_upsample_layers_; ++layer_idx) {
      const int32_t stride = cfg.upsample_rates[layer_idx];
      const int32_t kernel = cfg.upsample_kernel_sizes[layer_idx];
      const int32_t in_ch = cfg.upsample_initial_channel / static_cast<int32_t>(std::pow(2, layer_idx));
      const int32_t out_ch = cfg.upsample_initial_channel / static_cast<int32_t>(std::pow(2, layer_idx + 1));
      const int32_t padding = (kernel - stride) / 2;
      ups_.emplace_back(reg<nn::ConvTranspose1D>("ups." + std::to_string(layer_idx) + ".0", in_ch, out_ch, kernel, stride,
                                                padding, 0, 1, 1, true));
    }

    for (int32_t layer_idx = 0; layer_idx < num_upsample_layers_; ++layer_idx) {
      const int32_t channels = cfg.upsample_initial_channel / static_cast<int32_t>(std::pow(2, layer_idx + 1));
      for (size_t i = 0; i < cfg.resblock_kernel_sizes.size(); ++i) {
        resblocks_.emplace_back(reg<AMPBlock>("resblocks." + std::to_string(resblocks_.size()), channels,
                                              cfg.resblock_kernel_sizes[i], cfg.resblock_dilation_sizes[i]));
      }
    }

    activation_post_ =
        reg<TorchActivation1d>("activation_post", cfg.upsample_initial_channel / static_cast<int32_t>(std::pow(2, num_upsample_layers_)));
    conv_post_ = reg<nn::Conv1D>("conv_post",
                                cfg.upsample_initial_channel / static_cast<int32_t>(std::pow(2, num_upsample_layers_)), 1, 7, 1, 3, 1, 1,
                                false);
  }

  Tensor forward(const Tensor& mel_spectrogram) {
    auto mel = mel_spectrogram;
    if (!mel.isContiguous()) { mel = mel.contiguous(); }
    auto processed = processMelSpectrogram(mel);
    return forwardProcessed(processed);
  }

 private:
  Tensor forwardProcessed(const Tensor& processed) {
    auto hidden = conv_pre_(processed);

    for (int32_t layer_idx = 0; layer_idx < num_upsample_layers_; ++layer_idx) {
      hidden = ups_[layer_idx](hidden);
      Tensor residual_sum = Tensor::zeros(hidden.shape(), hidden.dtype(), hidden.device());
      for (int32_t block_idx = 0; block_idx < num_residual_blocks_; ++block_idx) {
        residual_sum = residual_sum + resblocks_[layer_idx * num_residual_blocks_ + block_idx].forward(hidden);
      }
      hidden = residual_sum * (1.0f / static_cast<float>(num_residual_blocks_));
    }

    hidden = activation_post_.forward({hidden}, {})[0];
    auto output = conv_post_(hidden);
    output = clampTensor(output, -1.0f, 1.0f);
    return output.squeeze();
  }
  Tensor processMelSpectrogram(const Tensor& mel_spectrogram) const {
    auto amplitude = nn::functional::exp(mel_spectrogram);
    auto decibel = amplitudeToDb(amplitude, -115.0f) + (-20.0f);
    return normalizeSpectrogram(decibel, 1.0f, -115.0f);
  }

  Qwen2_5OmniBigVGANConfig cfg_;
  int32_t num_residual_blocks_ = 0;
  int32_t num_upsample_layers_ = 0;
  nn::Conv1D conv_pre_;
  std::vector<nn::ConvTranspose1D> ups_;
  std::vector<AMPBlock> resblocks_;
  TorchActivation1d activation_post_;
  nn::Conv1D conv_post_;
};

class Qwen2_5OmniToken2WavModel final : public nn::Module {
 public:
  Qwen2_5OmniToken2WavModel() = default;
  explicit Qwen2_5OmniToken2WavModel(const std::string& name, const Qwen2_5OmniToken2WavConfig& cfg) : nn::Module(name), cfg_(cfg) {
    code2wav_dit_model_ = reg<Qwen2_5OmniToken2WavDiTModel>("code2wav_dit_model", cfg.dit_config);
    code2wav_bigvgan_model_ = reg<Qwen2_5OmniToken2WavBigVGANModel>("code2wav_bigvgan_model", cfg.bigvgan_config);
  }

  Tensor forward(const Tensor& code, const Tensor& conditioning, const Tensor& reference_mel, int32_t num_steps = 10,
                 float guidance_scale = 0.5f, float sway_coefficient = -1.0f) {
    auto mel = code2wav_dit_model_.sample(conditioning, reference_mel, code, num_steps, guidance_scale, sway_coefficient);
    if (!mel.isContiguous()) { mel = mel.contiguous(); }
    return code2wav_bigvgan_model_.forward(mel);
  }

  Tensor vocodeMel(const Tensor& mel) {
    return code2wav_bigvgan_model_.forward(mel);
  }

 private:
  Qwen2_5OmniToken2WavConfig cfg_;
  Qwen2_5OmniToken2WavDiTModel code2wav_dit_model_;
  Qwen2_5OmniToken2WavBigVGANModel code2wav_bigvgan_model_;
};

}  // namespace token2wav

using token2wav::Qwen2_5OmniToken2WavBigVGANModel;
using token2wav::Qwen2_5OmniToken2WavDiTModel;
using token2wav::Qwen2_5OmniToken2WavModel;

}  // namespace mllm::models::qwen2_5omni
