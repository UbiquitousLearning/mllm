// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "mllm/core/Parallel.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/layers/STFT.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/utils/Log.hpp"

#include "mllm/models/minicpm_o45/token2wav_prompt_cache.hpp"
#include "mllm/models/minicpm_o45/token2wav_weight_norm.hpp"

namespace mllm::models::minicpm_o45 {

struct MiniCPMO45FlowConfig {
  int32_t input_size = 512;
  int32_t output_size = 80;
  int32_t spk_embed_dim = 192;
  int32_t vocab_size = 6561;
  int32_t up_rate = 2;

  int32_t encoder_attention_heads = 8;
  int32_t encoder_linear_units = 2048;
  int32_t encoder_num_blocks = 6;
  int32_t encoder_num_up_blocks = 4;
  int32_t pre_lookahead_len = 3;

  int32_t dit_in_channels = 320;
  int32_t dit_out_channels = 80;
  float dit_mlp_ratio = 4.0f;
  int32_t dit_depth = 16;
  int32_t dit_num_heads = 8;
  int32_t dit_head_dim = 64;
  int32_t dit_hidden_size = 512;
  float cfm_inference_cfg_rate = 0.7f;
};

struct MiniCPMO45HiFTConfig {
  int32_t in_channels = 80;
  int32_t base_channels = 512;
  int32_t nb_harmonics = 8;
  int32_t sampling_rate = 24000;
  float nsf_alpha = 0.1f;
  float nsf_sigma = 0.003f;
  float nsf_voiced_threshold = 10.0f;
  std::vector<int32_t> upsample_rates = {8, 5, 3};
  std::vector<int32_t> upsample_kernel_sizes = {16, 11, 7};
  int32_t istft_n_fft = 16;
  int32_t istft_hop_len = 4;
  std::vector<int32_t> resblock_kernel_sizes = {3, 7, 11};
  std::vector<std::vector<int32_t>> resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  std::vector<int32_t> source_resblock_kernel_sizes = {7, 7, 11};
  std::vector<std::vector<int32_t>> source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  float lrelu_slope = 0.1f;
  float audio_limit = 0.99f;
};

struct MiniCPMO45Token2WavConfig {
  MiniCPMO45FlowConfig flow{};
  MiniCPMO45HiFTConfig hift{};
};

namespace token2wav {

inline bool isDebugEnabled() {
  static bool enabled = []() {
    const char* v = std::getenv("MLLM_TOKEN2WAV_DEBUG");
    if (v == nullptr) { return false; }
    return std::string(v) != "0";
  }();
  return enabled;
}

inline void debugLog(const std::string& msg) {
  if (!isDebugEnabled()) { return; }
  std::cerr << "[token2wav-cpp] " << msg << std::endl;
}

inline std::string shapeOf(const Tensor& x) {
  std::string s = "[";
  const auto& sh = x.shape();
  for (int32_t i = 0; i < static_cast<int32_t>(sh.size()); ++i) {
    s += std::to_string(sh[i]);
    if (i + 1 != static_cast<int32_t>(sh.size())) { s += ","; }
  }
  s += "]";
  return s;
}

inline std::string descOf(const Tensor& x) {
  return "shape=" + shapeOf(x) + ",dtype=" + std::to_string(static_cast<int32_t>(x.dtype()));
}

inline Tensor repeatInterleaveSeq(Tensor x, int32_t repeats) {
  MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);
  auto in = x.contiguous();
  const auto& shape = in.shape();
  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(shape.size()), 3);
  const int32_t batch = shape[0];
  const int32_t seq_len = shape[1];
  const int32_t channels = shape[2];

  auto out = Tensor::empty({batch, seq_len * repeats, channels}, kFloat32, kCPU).alloc();
  const auto* src = in.ptr<float>();
  auto* dst = out.ptr<float>();
  const int64_t in_stride_b = static_cast<int64_t>(seq_len) * channels;
  const int64_t out_stride_b = static_cast<int64_t>(seq_len) * repeats * channels;

  for (int32_t b = 0; b < batch; ++b) {
    const float* src_b = src + static_cast<int64_t>(b) * in_stride_b;
    float* dst_b = dst + static_cast<int64_t>(b) * out_stride_b;
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

inline Tensor concatInt64Seq(Tensor a, Tensor b) {
  MLLM_RT_ASSERT_EQ(a.dtype(), kInt64);
  MLLM_RT_ASSERT_EQ(b.dtype(), kInt64);
  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(a.shape().size()), 2);
  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(b.shape().size()), 2);
  MLLM_RT_ASSERT_EQ(a.shape()[0], b.shape()[0]);

  auto av = a.contiguous();
  auto bv = b.contiguous();
  const int32_t B = av.shape()[0];
  const int32_t Ta = av.shape()[1];
  const int32_t Tb = bv.shape()[1];
  auto out = Tensor::empty({B, Ta + Tb}, kInt64, kCPU).alloc();

  const auto* ap = av.ptr<int64_t>();
  const auto* bp = bv.ptr<int64_t>();
  auto* op = out.ptr<int64_t>();
  for (int32_t bidx = 0; bidx < B; ++bidx) {
    std::memcpy(op + static_cast<int64_t>(bidx) * (Ta + Tb), ap + static_cast<int64_t>(bidx) * Ta, sizeof(int64_t) * Ta);
    std::memcpy(op + static_cast<int64_t>(bidx) * (Ta + Tb) + Ta, bp + static_cast<int64_t>(bidx) * Tb, sizeof(int64_t) * Tb);
  }
  return out;
}

inline Tensor repeatInterleave1d(Tensor x, int32_t repeats) {
  MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);
  auto in = x.contiguous();
  const auto& shape = in.shape();
  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(shape.size()), 3);
  const int32_t batch = shape[0];
  const int32_t channels = shape[1];
  const int32_t seq_len = shape[2];

  auto out = Tensor::empty({batch, channels, seq_len * repeats}, kFloat32, kCPU).alloc();
  const auto* src = in.ptr<float>();
  auto* dst = out.ptr<float>();

  const int64_t in_stride_b = static_cast<int64_t>(channels) * seq_len;
  const int64_t out_stride_b = static_cast<int64_t>(channels) * seq_len * repeats;

  for (int32_t b = 0; b < batch; ++b) {
    const float* src_b = src + static_cast<int64_t>(b) * in_stride_b;
    float* dst_b = dst + static_cast<int64_t>(b) * out_stride_b;
    for (int32_t c = 0; c < channels; ++c) {
      const float* src_c = src_b + static_cast<int64_t>(c) * seq_len;
      float* dst_c = dst_b + static_cast<int64_t>(c) * seq_len * repeats;
      for (int32_t t = 0; t < seq_len; ++t) {
        const float v = src_c[t];
        for (int32_t r = 0; r < repeats; ++r) { dst_c[t * repeats + r] = v; }
      }
    }
  }
  return out;
}

inline Tensor l2NormalizeRow(Tensor x, float eps = 1e-12f) {
  MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);
  auto in = x.contiguous();
  const auto& shape = in.shape();
  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(shape.size()), 2);
  const int32_t batch = shape[0];
  const int32_t dim = shape[1];
  auto out = Tensor::empty(shape, kFloat32, kCPU).alloc();

  const auto* src = in.ptr<float>();
  auto* dst = out.ptr<float>();
  for (int32_t b = 0; b < batch; ++b) {
    const int64_t base = static_cast<int64_t>(b) * dim;
    float norm = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
      const float v = src[base + i];
      norm += v * v;
    }
    norm = std::sqrt(std::max(norm, eps));
    for (int32_t i = 0; i < dim; ++i) { dst[base + i] = src[base + i] / norm; }
  }
  return out;
}

inline Tensor tensorMish(Tensor x) {
  MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);
  auto in = x.contiguous();
  auto out = Tensor::empty(in.shape(), kFloat32, kCPU).alloc();
  const auto* src = in.ptr<float>();
  auto* dst = out.ptr<float>();
  const int64_t n = static_cast<int64_t>(in.numel());
  MLLM_CONDITIONAL_PARALLEL_FOR(n > 4096, 4, i, 0, n, 1, {
    const float v = src[i];
    const float sp = std::log1p(std::exp(v));
    dst[i] = v * std::tanh(sp);
  });
  return out;
}

inline Tensor tensorLeakyRelu(Tensor x, float slope) {
  MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);
  auto in = x.contiguous();
  auto out = Tensor::empty(in.shape(), kFloat32, kCPU).alloc();
  const auto* src = in.ptr<float>();
  auto* dst = out.ptr<float>();
  const int64_t n = static_cast<int64_t>(in.numel());
  MLLM_CONDITIONAL_PARALLEL_FOR(n > 4096, 4, i, 0, n, 1, {
    const float v = src[i];
    dst[i] = (v >= 0.0f) ? v : (v * slope);
  });
  return out;
}

inline Tensor makeHannWindow(int32_t win_length) {
  auto w = Tensor::empty({1, win_length}, kFloat32, kCPU).alloc();
  auto* ptr = w.ptr<float>();
  constexpr float kPi = 3.14159265358979323846f;
  for (int32_t i = 0; i < win_length; ++i) {
    ptr[i] = 0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(win_length));
  }
  return w;
}

inline Tensor relShift(Tensor x) {
  // x: [B, H, T, 2T-1], output [B, H, T, T]
  MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);
  auto in = x.contiguous();
  const int32_t B = in.shape()[0];
  const int32_t H = in.shape()[1];
  const int32_t T = in.shape()[2];
  const int32_t R = in.shape()[3];
  MLLM_RT_ASSERT_EQ(R, 2 * T - 1);

  auto out = Tensor::empty({B, H, T, T}, kFloat32, kCPU).alloc();
  const auto* src = in.ptr<float>();
  auto* dst = out.ptr<float>();
  const int64_t in_stride_b = static_cast<int64_t>(H) * T * R;
  const int64_t in_stride_h = static_cast<int64_t>(T) * R;
  const int64_t in_stride_t = R;
  const int64_t out_stride_b = static_cast<int64_t>(H) * T * T;
  const int64_t out_stride_h = static_cast<int64_t>(T) * T;
  const int64_t out_stride_t = T;

  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      const float* src_h = src + static_cast<int64_t>(b) * in_stride_b + static_cast<int64_t>(h) * in_stride_h;
      float* dst_h = dst + static_cast<int64_t>(b) * out_stride_b + static_cast<int64_t>(h) * out_stride_h;
      for (int32_t i = 0; i < T; ++i) {
        const float* src_i = src_h + static_cast<int64_t>(i) * in_stride_t;
        float* dst_i = dst_h + static_cast<int64_t>(i) * out_stride_t;
        for (int32_t j = 0; j < T; ++j) {
          const int32_t src_idx = j - i + T - 1;
          dst_i[j] = src_i[src_idx];
        }
      }
    }
  }
  return out;
}

inline void addHeadBiasInplace(Tensor& q, Tensor bias) {
  // q: [B, H, T, D], bias: [H, D]
  MLLM_RT_ASSERT_EQ(q.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(bias.dtype(), kFloat32);
  auto qv = q.contiguous();
  auto bv = bias.contiguous();
  const int32_t B = qv.shape()[0];
  const int32_t H = qv.shape()[1];
  const int32_t T = qv.shape()[2];
  const int32_t D = qv.shape()[3];
  MLLM_RT_ASSERT_EQ(bv.shape()[0], H);
  MLLM_RT_ASSERT_EQ(bv.shape()[1], D);
  auto* q_ptr = qv.ptr<float>();
  const auto* b_ptr = bv.ptr<float>();

  const int64_t q_stride_b = static_cast<int64_t>(H) * T * D;
  const int64_t q_stride_h = static_cast<int64_t>(T) * D;
  const int64_t q_stride_t = D;

  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      const float* bh = b_ptr + static_cast<int64_t>(h) * D;
      for (int32_t t = 0; t < T; ++t) {
        float* row = q_ptr + static_cast<int64_t>(b) * q_stride_b + static_cast<int64_t>(h) * q_stride_h
                     + static_cast<int64_t>(t) * q_stride_t;
        for (int32_t d = 0; d < D; ++d) { row[d] += bh[d]; }
      }
    }
  }
  q = qv;
}

inline Tensor concatChannel(const std::vector<Tensor>& xs) {
  return nn::functional::concat(xs, 1);
}

inline Tensor makeTimeStepsTensor(const std::vector<float>& values) {
  auto t = Tensor::empty({static_cast<int32_t>(values.size())}, kFloat32, kCPU).alloc();
  auto* ptr = t.ptr<float>();
  for (size_t i = 0; i < values.size(); ++i) { ptr[i] = values[i]; }
  return t;
}

inline Tensor randomNormalLike(const std::vector<int32_t>& shape, float scale = 1.0f) {
  auto out = Tensor::empty(shape, kFloat32, kCPU).alloc();
  auto* ptr = out.ptr<float>();
  const int64_t n = static_cast<int64_t>(out.numel());
  static thread_local std::mt19937 rng(std::random_device{}());
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (int64_t i = 0; i < n; ++i) { ptr[i] = dist(rng) * scale; }
  return out;
}

class EspnetRelPositionalEncoding final : public nn::Module {
 public:
  EspnetRelPositionalEncoding() = default;
  EspnetRelPositionalEncoding(const std::string& name, int32_t dim) : nn::Module(name), dim_(dim) { xscale_ = std::sqrt(static_cast<float>(dim)); }

  std::pair<Tensor, Tensor> forwardWithPos(Tensor x) {
    const int32_t T = x.shape()[1];
    auto pos = positionEncoding(T);
    return {x * xscale_, pos};
  }

 private:
  Tensor positionEncoding(int32_t size) const {
    const int32_t dim = dim_;
    auto pe_pos = Tensor::empty({size, dim}, kFloat32, kCPU).alloc();
    auto pe_neg = Tensor::empty({size, dim}, kFloat32, kCPU).alloc();
    auto* pos_ptr = pe_pos.ptr<float>();
    auto* neg_ptr = pe_neg.ptr<float>();

    for (int32_t p = 0; p < size; ++p) {
      for (int32_t i = 0; i < dim; i += 2) {
        const float div = std::exp(-std::log(10000.0f) * static_cast<float>(i) / static_cast<float>(dim));
        const float v1 = std::sin(static_cast<float>(p) * div);
        const float v2 = std::cos(static_cast<float>(p) * div);
        pos_ptr[p * dim + i] = v1;
        pos_ptr[p * dim + i + 1] = v2;
        neg_ptr[p * dim + i] = -v1;
        neg_ptr[p * dim + i + 1] = v2;
      }
    }

    auto pe_positive = Tensor::empty({1, size, dim}, kFloat32, kCPU).alloc();
    auto pe_negative = Tensor::empty({1, std::max(size - 1, 0), dim}, kFloat32, kCPU).alloc();
    auto* pp = pe_positive.ptr<float>();
    auto* pn = pe_negative.ptr<float>();
    for (int32_t i = 0; i < size; ++i) {
      std::memcpy(pp + static_cast<int64_t>(i) * dim, pos_ptr + static_cast<int64_t>(size - 1 - i) * dim,
                  sizeof(float) * dim);
    }
    for (int32_t i = 1; i < size; ++i) {
      std::memcpy(pn + static_cast<int64_t>(i - 1) * dim, neg_ptr + static_cast<int64_t>(i) * dim, sizeof(float) * dim);
    }
    return nn::functional::concat({pe_positive, pe_negative}, 1);
  }

 private:
  int32_t dim_ = 0;
  float xscale_ = 1.0f;
};

class LinearNoSubsampling final : public nn::Module {
 public:
  LinearNoSubsampling() = default;
  LinearNoSubsampling(const std::string& name, int32_t idim, int32_t odim) : nn::Module(name) {
    out_linear_ = reg<nn::Linear>("out.0", idim, odim, true);
    out_norm_ = reg<nn::LayerNorm>("out.1", std::vector<int32_t>{odim}, true, true, 1e-5f);
    pos_enc_ = reg<EspnetRelPositionalEncoding>("pos_enc", odim);
  }

  std::pair<Tensor, Tensor> forwardWithPos(Tensor x) {
    auto y = out_linear_(x);
    y = out_norm_(y);
    return pos_enc_.forwardWithPos(y);
  }

 private:
  nn::Linear out_linear_;
  nn::LayerNorm out_norm_;
  EspnetRelPositionalEncoding pos_enc_;
};

class PositionwiseFeedForward final : public nn::Module {
 public:
  PositionwiseFeedForward() = default;
  PositionwiseFeedForward(const std::string& name, int32_t idim, int32_t hidden_units) : nn::Module(name) {
    w1_ = reg<nn::Linear>("w_1", idim, hidden_units, true);
    w2_ = reg<nn::Linear>("w_2", hidden_units, idim, true);
  }

  Tensor forwardOne(Tensor x) {
    auto y = w1_(x);
    y = nn::functional::silu(y);
    y = w2_(y);
    return y;
  }

 private:
  nn::Linear w1_;
  nn::Linear w2_;
};

class RelPositionMultiHeadedAttention final : public nn::Module {
 public:
  RelPositionMultiHeadedAttention() = default;
  RelPositionMultiHeadedAttention(const std::string& name, int32_t n_head, int32_t n_feat, bool key_bias)
      : nn::Module(name), n_head_(n_head), n_feat_(n_feat) {
    d_k_ = n_feat_ / n_head_;
    linear_q_ = reg<nn::Linear>("linear_q", n_feat_, n_feat_, true);
    linear_k_ = reg<nn::Linear>("linear_k", n_feat_, n_feat_, key_bias);
    linear_v_ = reg<nn::Linear>("linear_v", n_feat_, n_feat_, true);
    linear_out_ = reg<nn::Linear>("linear_out", n_feat_, n_feat_, true);
    linear_pos_ = reg<nn::Linear>("linear_pos", n_feat_, n_feat_, false);
    pos_bias_u_ = reg<nn::Param>("pos_bias_u", getModuleName() + ".pos_bias_u", Tensor::shape_t{n_head_, d_k_});
    pos_bias_v_ = reg<nn::Param>("pos_bias_v", getModuleName() + ".pos_bias_v", Tensor::shape_t{n_head_, d_k_});
  }

  Tensor forwardOne(Tensor x, Tensor pos_emb) {
    auto q = linear_q_(x).view({x.shape()[0], x.shape()[1], n_head_, d_k_}).transpose(1, 2);  // [B,H,T,D]
    auto k = linear_k_(x).view({x.shape()[0], x.shape()[1], n_head_, d_k_}).transpose(1, 2);
    auto v = linear_v_(x).view({x.shape()[0], x.shape()[1], n_head_, d_k_}).transpose(1, 2);

    auto p = linear_pos_(pos_emb).view({pos_emb.shape()[0], pos_emb.shape()[1], n_head_, d_k_}).transpose(1, 2);  // [1,H,2T-1,D]

    auto q_with_bias_u = q.contiguous();
    auto q_with_bias_v = q.contiguous();
    addHeadBiasInplace(q_with_bias_u, pos_bias_u_.weight());
    addHeadBiasInplace(q_with_bias_v, pos_bias_v_.weight());

    auto matrix_ac = nn::functional::matmul(q_with_bias_u, k.transpose(2, 3), false, false);  // [B,H,T,T]
    auto matrix_bd = nn::functional::matmul(q_with_bias_v, p.transpose(2, 3), false, false);  // [B,H,T,2T-1]
    if (matrix_ac.shape()[3] != matrix_bd.shape()[3]) { matrix_bd = relShift(matrix_bd); }
    auto scores = (matrix_ac + matrix_bd) / std::sqrt(static_cast<float>(d_k_));
    auto attn = nn::functional::softmax(scores, -1);
    auto y = nn::functional::matmul(attn, v, false, false);  // [B,H,T,D]
    y = y.transpose(1, 2).view({x.shape()[0], x.shape()[1], n_feat_});
    return linear_out_(y);
  }

 private:
  int32_t n_head_ = 0;
  int32_t n_feat_ = 0;
  int32_t d_k_ = 0;
  nn::Linear linear_q_;
  nn::Linear linear_k_;
  nn::Linear linear_v_;
  nn::Linear linear_out_;
  nn::Linear linear_pos_;
  nn::Param pos_bias_u_;
  nn::Param pos_bias_v_;
};

class ConformerEncoderLayer final : public nn::Module {
 public:
  ConformerEncoderLayer() = default;
  ConformerEncoderLayer(const std::string& name, int32_t size, int32_t n_head, int32_t linear_units, bool key_bias)
      : nn::Module(name) {
    self_attn_ = reg<RelPositionMultiHeadedAttention>("self_attn", n_head, size, key_bias);
    feed_forward_ = reg<PositionwiseFeedForward>("feed_forward", size, linear_units);
    norm_ff_ = reg<nn::LayerNorm>("norm_ff", std::vector<int32_t>{size}, true, true, 1e-12f);
    norm_mha_ = reg<nn::LayerNorm>("norm_mha", std::vector<int32_t>{size}, true, true, 1e-12f);
  }

  Tensor forwardOne(Tensor x, Tensor pos_emb) {
    auto h = norm_mha_(x);
    auto y = self_attn_.forwardOne(h, pos_emb);
    y = x + y;
    auto z = norm_ff_(y);
    z = feed_forward_.forwardOne(z);
    return y + z;
  }

 private:
  RelPositionMultiHeadedAttention self_attn_;
  PositionwiseFeedForward feed_forward_;
  nn::LayerNorm norm_ff_;
  nn::LayerNorm norm_mha_;
};

class PreLookaheadLayer final : public nn::Module {
 public:
  PreLookaheadLayer() = default;
  PreLookaheadLayer(const std::string& name, int32_t channels, int32_t pre_lookahead_len) : nn::Module(name), pre_(pre_lookahead_len) {
    conv1_ = reg<nn::Conv1D>("conv1", channels, channels, pre_ + 1, 1, 0, 1, 1, true);
    conv2_ = reg<nn::Conv1D>("conv2", channels, channels, 3, 1, 0, 1, 1, true);
  }

  Tensor forwardOne(Tensor inputs) {
    auto x = inputs.transpose(1, 2).contiguous();                               // [B,C,T]
    x = nn::functional::pad(x, {0, pre_}, aops::PadMode::kConstant, 0.0f);     // right pad
    x = conv1_(x);
    x = tensorLeakyRelu(x, 0.01f);
    x = nn::functional::pad(x, {2, 0}, aops::PadMode::kConstant, 0.0f);        // left pad
    x = conv2_(x);
    x = x.transpose(1, 2).contiguous();                                          // [B,T,C]
    return x + inputs;
  }

 private:
  int32_t pre_ = 3;
  nn::Conv1D conv1_;
  nn::Conv1D conv2_;
};

class Upsample1D final : public nn::Module {
 public:
  Upsample1D() = default;
  Upsample1D(const std::string& name, int32_t channels, int32_t out_channels, int32_t stride) : nn::Module(name), stride_(stride) {
    conv_ = reg<nn::Conv1D>("conv", channels, out_channels, stride_ * 2 + 1, 1, 0, 1, 1, true);
  }

  Tensor forwardOne(Tensor inputs) {
    auto x = repeatInterleave1d(inputs, stride_);
    x = nn::functional::pad(x, {stride_ * 2, 0}, aops::PadMode::kConstant, 0.0f);
    return conv_(x);
  }

  int32_t stride() const { return stride_; }

 private:
  int32_t stride_ = 2;
  nn::Conv1D conv_;
};

class UpsampleConformerEncoderV2 final : public nn::Module {
 public:
  UpsampleConformerEncoderV2() = default;
  UpsampleConformerEncoderV2(const std::string& name, const MiniCPMO45FlowConfig& cfg) : nn::Module(name), cfg_(cfg) {
    embed_ = reg<LinearNoSubsampling>("embed", cfg.input_size, cfg.input_size);
    pre_lookahead_ = reg<PreLookaheadLayer>("pre_lookahead_layer", cfg.input_size, cfg.pre_lookahead_len);
    encoders_ = reg<nn::ModuleList<ConformerEncoderLayer>>("encoders", cfg.encoder_num_blocks, cfg.input_size,
                                                            cfg.encoder_attention_heads, cfg.encoder_linear_units, true);
    up_layer_ = reg<Upsample1D>("up_layer", cfg.input_size, cfg.input_size, cfg.up_rate);
    up_embed_ = reg<LinearNoSubsampling>("up_embed", cfg.input_size, cfg.input_size);
    up_encoders_ = reg<nn::ModuleList<ConformerEncoderLayer>>("up_encoders", cfg.encoder_num_up_blocks, cfg.input_size,
                                                               cfg.encoder_attention_heads, cfg.encoder_linear_units, true);
    after_norm_ = reg<nn::LayerNorm>("after_norm", std::vector<int32_t>{cfg.input_size}, true, true, 1e-5f);
  }

  Tensor forwardOne(Tensor xs) {
    auto [x0, pos0] = embed_.forwardWithPos(xs);
    x0 = pre_lookahead_.forwardOne(x0);
    for (auto& layer : encoders_.list()) { x0 = layer.forwardOne(x0, pos0); }

    x0 = x0.transpose(1, 2).contiguous();
    x0 = up_layer_.forwardOne(x0);
    x0 = x0.transpose(1, 2).contiguous();

    auto [x1, pos1] = up_embed_.forwardWithPos(x0);
    for (auto& layer : up_encoders_.list()) { x1 = layer.forwardOne(x1, pos1); }
    x1 = after_norm_(x1);
    return x1;
  }

 private:
  MiniCPMO45FlowConfig cfg_;
  LinearNoSubsampling embed_;
  PreLookaheadLayer pre_lookahead_;
  nn::ModuleList<ConformerEncoderLayer> encoders_;
  Upsample1D up_layer_;
  LinearNoSubsampling up_embed_;
  nn::ModuleList<ConformerEncoderLayer> up_encoders_;
  nn::LayerNorm after_norm_;
};

class DiTAttention final : public nn::Module {
 public:
  DiTAttention() = default;
  DiTAttention(const std::string& name, int32_t dim, int32_t num_heads, int32_t head_dim) : nn::Module(name),
      dim_(dim), heads_(num_heads), head_dim_(head_dim), inner_dim_(num_heads * head_dim) {
    to_q_ = reg<nn::Linear>("to_q", dim_, inner_dim_, true);
    to_k_ = reg<nn::Linear>("to_k", dim_, inner_dim_, true);
    to_v_ = reg<nn::Linear>("to_v", dim_, inner_dim_, true);
    q_norm_ = reg<nn::LayerNorm>("q_norm", std::vector<int32_t>{head_dim_}, true, true, 1e-5f);
    k_norm_ = reg<nn::LayerNorm>("k_norm", std::vector<int32_t>{head_dim_}, true, true, 1e-5f);
    proj_ = reg<nn::Linear>("proj", inner_dim_, dim_, true);
  }

  Tensor forwardOne(Tensor x) {
    debugLog("dit.attn: enter x(" + descOf(x) + ")");
    auto q = to_q_(x).view({x.shape()[0], x.shape()[1], heads_, head_dim_}).transpose(1, 2);  // [B,H,T,D]
    debugLog("dit.attn: to_q done");
    auto k = to_k_(x).view({x.shape()[0], x.shape()[1], heads_, head_dim_}).transpose(1, 2);
    debugLog("dit.attn: to_k done");
    auto v = to_v_(x).view({x.shape()[0], x.shape()[1], heads_, head_dim_}).transpose(1, 2);
    debugLog("dit.attn: to_v done");

    q = q_norm_(q);
    k = k_norm_(k);

    auto out = nn::functional::scaledDotProductAttention(q, k, v);  // [B,H,T,D]
    out = out.transpose(1, 2).contiguous().view({x.shape()[0], x.shape()[1], inner_dim_});
    out = proj_(out);
    debugLog("dit.attn: exit");
    return out;
  }

 private:
  int32_t dim_ = 0;
  int32_t heads_ = 0;
  int32_t head_dim_ = 0;
  int32_t inner_dim_ = 0;
  nn::Linear to_q_;
  nn::Linear to_k_;
  nn::Linear to_v_;
  nn::LayerNorm q_norm_;
  nn::LayerNorm k_norm_;
  nn::Linear proj_;
};

class CausalConv1dBlock final : public nn::Module {
 public:
  CausalConv1dBlock() = default;
  CausalConv1dBlock(const std::string& name, int32_t in_channels, int32_t out_channels, int32_t kernel_size) : nn::Module(name),
      in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size) {
    conv1_ = reg<nn::Conv1D>("block.1", in_channels_, out_channels_, kernel_size_, 1, 0, 1, 1, true);
    norm_ = reg<nn::LayerNorm>("block.3", std::vector<int32_t>{out_channels_}, true, true, 1e-5f);
    conv2_ = reg<nn::Conv1D>("block.6", out_channels_, out_channels_, kernel_size_, 1, 0, 1, 1, true);
  }

  Tensor forwardOne(Tensor x) {
    auto y = x.transpose(1, 2).contiguous();
    y = nn::functional::pad(y, {kernel_size_ - 1, 0}, aops::PadMode::kConstant, 0.0f);
    y = conv1_(y);
    y = y.transpose(1, 2).contiguous();
    y = norm_(y);
    y = tensorMish(y);
    y = y.transpose(1, 2).contiguous();
    y = nn::functional::pad(y, {kernel_size_ - 1, 0}, aops::PadMode::kConstant, 0.0f);
    y = conv2_(y);
    y = y.transpose(1, 2).contiguous();
    return y;
  }

 private:
  int32_t in_channels_ = 0;
  int32_t out_channels_ = 0;
  int32_t kernel_size_ = 3;
  nn::Conv1D conv1_;
  nn::LayerNorm norm_;
  nn::Conv1D conv2_;
};

class DiTMLP final : public nn::Module {
 public:
  DiTMLP() = default;
  DiTMLP(const std::string& name, int32_t in_features, int32_t hidden_features) : nn::Module(name) {
    fc1_ = reg<nn::Linear>("fc1", in_features, hidden_features, true);
    gelu_ = reg<nn::GELU>("act");
    fc2_ = reg<nn::Linear>("fc2", hidden_features, in_features, true);
  }

  Tensor forwardOne(Tensor x) {
    auto y = fc1_(x);
    y = gelu_(y);
    y = fc2_(y);
    return y;
  }

 private:
  nn::Linear fc1_;
  nn::GELU gelu_;
  nn::Linear fc2_;
};

class TimestepEmbedder final : public nn::Module {
 public:
  TimestepEmbedder() = default;
  TimestepEmbedder(const std::string& name, int32_t hidden_size, int32_t frequency_embedding_size = 256)
      : nn::Module(name), hidden_size_(hidden_size), freq_size_(frequency_embedding_size) {
    fc1_ = reg<nn::Linear>("mlp.0", freq_size_, hidden_size_, true);
    act_ = reg<nn::SiLU>("mlp.1");
    fc2_ = reg<nn::Linear>("mlp.2", hidden_size_, hidden_size_, true);
  }

  Tensor forwardOne(Tensor t) {
    auto emb = timestepEmbedding(t, freq_size_);
    emb = fc1_(emb);
    emb = act_(emb);
    emb = fc2_(emb);
    return emb;
  }

 private:
  Tensor timestepEmbedding(Tensor t, int32_t dim) const {
    MLLM_RT_ASSERT_EQ(t.dtype(), kFloat32);
    auto tt = t.contiguous();
    const int32_t N = tt.shape()[0];
    const int32_t half = dim / 2;
    auto out = Tensor::empty({N, dim}, kFloat32, kCPU).alloc();
    const auto* tp = tt.ptr<float>();
    auto* op = out.ptr<float>();
    for (int32_t i = 0; i < N; ++i) {
      const float tv = tp[i] * 1000.0f;
      for (int32_t j = 0; j < half; ++j) {
        const float freq = std::exp(-std::log(10000.0f) * static_cast<float>(j) / static_cast<float>(half));
        const float a = tv * freq;
        op[i * dim + j] = std::cos(a);
        op[i * dim + half + j] = std::sin(a);
      }
      if (dim % 2 == 1) { op[i * dim + dim - 1] = 0.0f; }
    }
    return out;
  }

 private:
  int32_t hidden_size_ = 0;
  int32_t freq_size_ = 0;
  nn::Linear fc1_;
  nn::SiLU act_;
  nn::Linear fc2_;
};

class FinalLayer final : public nn::Module {
 public:
  FinalLayer() = default;
  FinalLayer(const std::string& name, int32_t hidden_size, int32_t out_channels) : nn::Module(name) {
    adaln_act_ = reg<nn::SiLU>("adaLN_modulation.0");
    adaln_linear_ = reg<nn::Linear>("adaLN_modulation.1", hidden_size, 2 * hidden_size, true);
    norm_ = reg<nn::LayerNorm>("norm_final", std::vector<int32_t>{hidden_size}, false, false, 1e-6f);
    linear_ = reg<nn::Linear>("linear", hidden_size, out_channels, true);
  }

  Tensor forwardOne(Tensor x, Tensor c) {
    auto m = adaln_linear_(adaln_act_(c));
    auto chunks = nn::functional::chunk<2>(m, 2);
    auto shift = chunks[0];
    auto scale = chunks[1];
    auto y = norm_(x);
    if (scale.rank() == 2) { scale = scale.view({scale.shape()[0], 1, scale.shape()[1]}); }
    if (shift.rank() == 2) { shift = shift.view({shift.shape()[0], 1, shift.shape()[1]}); }
    y = y * (scale + 1.0f) + shift;
    y = linear_(y);
    return y;
  }

 private:
  nn::SiLU adaln_act_;
  nn::Linear adaln_linear_;
  nn::LayerNorm norm_;
  nn::Linear linear_;
};

class DiTBlock final : public nn::Module {
 public:
  DiTBlock() = default;
  DiTBlock(const std::string& name, int32_t hidden_size, int32_t num_heads, int32_t head_dim, float mlp_ratio)
      : nn::Module(name), hidden_size_(hidden_size) {
    norm1_ = reg<nn::LayerNorm>("norm1", std::vector<int32_t>{hidden_size_}, false, false, 1e-6f);
    attn_ = reg<DiTAttention>("attn", hidden_size_, num_heads, head_dim);
    norm2_ = reg<nn::LayerNorm>("norm2", std::vector<int32_t>{hidden_size_}, false, false, 1e-6f);
    norm3_ = reg<nn::LayerNorm>("norm3", std::vector<int32_t>{hidden_size_}, false, false, 1e-6f);
    const int32_t mlp_hidden = static_cast<int32_t>(hidden_size_ * mlp_ratio);
    mlp_ = reg<DiTMLP>("mlp", hidden_size_, mlp_hidden);
    conv_ = reg<CausalConv1dBlock>("conv", hidden_size_, hidden_size_, 3);
    adaln_act_ = reg<nn::SiLU>("adaLN_modulation.0");
    adaln_linear_ = reg<nn::Linear>("adaLN_modulation.1", hidden_size_, hidden_size_ * 9, true);
  }

  Tensor forwardOne(Tensor x, Tensor c) {
    debugLog("dit.block: enter x(" + descOf(x) + ") c(" + descOf(c) + ")");
    auto mods = adaln_linear_(adaln_act_(c));  // [B,1,9C]
    debugLog("dit.block: adaln_linear done mods(" + descOf(mods) + ")");
    const int32_t C = hidden_size_;
    auto shift_msa = mods[{kAll, kAll, {0 * C, 1 * C}}].contiguous();
    auto scale_msa = mods[{kAll, kAll, {1 * C, 2 * C}}].contiguous();
    auto gate_msa = mods[{kAll, kAll, {2 * C, 3 * C}}].contiguous();
    auto shift_mlp = mods[{kAll, kAll, {3 * C, 4 * C}}].contiguous();
    auto scale_mlp = mods[{kAll, kAll, {4 * C, 5 * C}}].contiguous();
    auto gate_mlp = mods[{kAll, kAll, {5 * C, 6 * C}}].contiguous();
    auto shift_conv = mods[{kAll, kAll, {6 * C, 7 * C}}].contiguous();
    auto scale_conv = mods[{kAll, kAll, {7 * C, 8 * C}}].contiguous();
    auto gate_conv = mods[{kAll, kAll, {8 * C, 9 * C}}].contiguous();
    debugLog("dit.block: chunk9 done");

    auto y = norm1_(x);
    y = y * (scale_msa + 1.0f) + shift_msa;
    debugLog("dit.block: before attn y(" + descOf(y) + ")");
    auto attn_out = attn_.forwardOne(y);
    debugLog("dit.block: attn done");
    auto h = x + attn_out * gate_msa;

    auto c_in = norm3_(h);
    c_in = c_in * (scale_conv + 1.0f) + shift_conv;
    auto conv_out = conv_.forwardOne(c_in);
    debugLog("dit.block: conv done");
    h = h + conv_out * gate_conv;

    auto m_in = norm2_(h);
    m_in = m_in * (scale_mlp + 1.0f) + shift_mlp;
    auto mlp_out = mlp_.forwardOne(m_in);
    debugLog("dit.block: mlp done");
    h = h + mlp_out * gate_mlp;
    debugLog("dit.block: exit");
    return h;
  }

 private:
  int32_t hidden_size_ = 0;
  nn::LayerNorm norm1_;
  DiTAttention attn_;
  nn::LayerNorm norm2_;
  nn::LayerNorm norm3_;
  DiTMLP mlp_;
  CausalConv1dBlock conv_;
  nn::SiLU adaln_act_;
  nn::Linear adaln_linear_;
};

class DiTEstimator final : public nn::Module {
 public:
  DiTEstimator() = default;
  DiTEstimator(const std::string& name, const MiniCPMO45FlowConfig& cfg) : nn::Module(name), cfg_(cfg) {
    t_embedder_ = reg<TimestepEmbedder>("t_embedder", cfg.dit_hidden_size, 256);
    in_proj_ = reg<nn::Linear>("in_proj", cfg.dit_in_channels, cfg.dit_hidden_size, true);
    blocks_ = reg<nn::ModuleList<DiTBlock>>("blocks", cfg.dit_depth, cfg.dit_hidden_size, cfg.dit_num_heads, cfg.dit_head_dim,
                                             cfg.dit_mlp_ratio);
    final_layer_ = reg<FinalLayer>("final_layer", cfg.dit_hidden_size, cfg.dit_out_channels);
  }

  Tensor forwardOne(Tensor x, Tensor mu, Tensor t, Tensor spks, Tensor cond) {
    // x,mu,cond: [B,C,T], spks: [B,C], t:[B]
    debugLog("dit.forward: begin");
    auto time_emb = t_embedder_.forwardOne(t).view({t.shape()[0], 1, cfg_.dit_hidden_size});
    debugLog("dit.forward: t_embedder done");
    auto spk_seq = spks.view({spks.shape()[0], spks.shape()[1], 1}).repeat(x.shape()[2], 2);
    auto packed = concatChannel({x, mu, spk_seq, cond});       // [B,320,T]
    debugLog("dit.forward: concat packed done");
    auto h = packed.transpose(1, 2).contiguous();              // [B,T,320]
    h = in_proj_(h);                                            // [B,T,512]
    debugLog("dit.forward: in_proj done");
    int32_t block_idx = 0;
    for (auto& block : blocks_.list()) {
      h = block.forwardOne(h, time_emb);
      if (block_idx == 0) { debugLog("dit.forward: block0 done"); }
      ++block_idx;
    }
    h = final_layer_.forwardOne(h, time_emb);                  // [B,T,80]
    debugLog("dit.forward: final_layer done");
    h = h.transpose(1, 2).contiguous();                        // [B,80,T]
    debugLog("dit.forward: end");
    return h;
  }

 private:
  MiniCPMO45FlowConfig cfg_;
  TimestepEmbedder t_embedder_;
  nn::Linear in_proj_;
  nn::ModuleList<DiTBlock> blocks_;
  FinalLayer final_layer_;
};

class CausalConditionalCFM final : public nn::Module {
 public:
  CausalConditionalCFM() = default;
  CausalConditionalCFM(const std::string& name, const MiniCPMO45FlowConfig& cfg) : nn::Module(name), cfg_(cfg) {
    estimator_ = reg<DiTEstimator>("estimator", cfg_);
  }

  Tensor forwardOne(Tensor mu, Tensor spks, Tensor cond, int32_t n_timesteps, float temperature = 1.0f) {
    // all in float32 cpu.
    debugLog("cfm.forward: start");
    const int32_t B = mu.shape()[0];
    const int32_t C = mu.shape()[1];
    const int32_t T = mu.shape()[2];
    MLLM_RT_ASSERT_EQ(B, 1);

    auto z = randomNormalLike({B, C, T}, temperature);

    std::vector<float> t_span(static_cast<size_t>(n_timesteps + 1), 0.0f);
    constexpr float kPi = 3.14159265358979323846f;
    for (int32_t i = 0; i <= n_timesteps; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(n_timesteps);
      t_span[static_cast<size_t>(i)] = 1.0f - std::cos(t * 0.5f * kPi);
    }

    auto x = z;
    auto mu_in = nn::functional::concat({mu, Tensor::zeros(mu.shape(), kFloat32, kCPU)}, 0);
    auto spk_in = nn::functional::concat({spks, Tensor::zeros(spks.shape(), kFloat32, kCPU)}, 0);
    auto cond_in = nn::functional::concat({cond, Tensor::zeros(cond.shape(), kFloat32, kCPU)}, 0);

    float t = t_span[0];
    float dt = t_span[1] - t_span[0];
    for (int32_t step = 1; step <= n_timesteps; ++step) {
      if (step == 1) { debugLog("cfm.forward: first estimator step"); }
      auto x_in = nn::functional::concat({x, x}, 0);  // [2,C,T]
      auto t_in = makeTimeStepsTensor({t, t});
      auto dphi = estimator_.forwardOne(x_in, mu_in, t_in, spk_in, cond_in);  // [2,C,T]
      auto dphi_split = nn::functional::chunk<2>(dphi, 0);
      auto dphi_main = dphi_split[0];
      auto dphi_cfg = dphi_split[1];
      auto dphi_out = dphi_main * (1.0f + cfg_.cfm_inference_cfg_rate) - dphi_cfg * cfg_.cfm_inference_cfg_rate;
      x = x + dphi_out * dt;
      t += dt;
      if (step < n_timesteps) { dt = t_span[static_cast<size_t>(step + 1)] - t; }
    }
    debugLog("cfm.forward: finish");
    return x;
  }

 private:
  MiniCPMO45FlowConfig cfg_;
  DiTEstimator estimator_;
};

class CausalMaskedDiffWithXvec final : public nn::Module {
 public:
  CausalMaskedDiffWithXvec() = default;
  CausalMaskedDiffWithXvec(const std::string& name, const MiniCPMO45FlowConfig& cfg) : nn::Module(name), cfg_(cfg) {
    input_embedding_ = reg<nn::Embedding>("input_embedding", cfg.vocab_size, cfg.input_size);
    spk_embed_affine_layer_ = reg<nn::Linear>("spk_embed_affine_layer", cfg.spk_embed_dim, cfg.output_size, true);
    encoder_ = reg<UpsampleConformerEncoderV2>("encoder", cfg);
    encoder_proj_ = reg<nn::Linear>("encoder_proj", cfg.input_size, cfg.output_size, true);
    decoder_ = reg<CausalConditionalCFM>("decoder", cfg);
  }

  Tensor inference(Tensor token, Tensor prompt_token, Tensor prompt_feat, Tensor embedding,
                   int32_t n_timesteps) {
    // token/prompt_token: [1,T], int64
    debugLog("flow.inference: start");
    auto spk = l2NormalizeRow(embedding);
    spk = spk_embed_affine_layer_(spk);  // [1,80]
    debugLog("flow.inference: spk_embed_affine_layer done");

    auto all_token = concatInt64Seq(prompt_token, token);
    auto token_embed = input_embedding_(all_token);
    debugLog("flow.inference: input_embedding done");

    auto h = encoder_.forwardOne(token_embed);
    debugLog("flow.inference: encoder done");
    h = encoder_proj_(h);  // [1, Tm, 80]
    debugLog("flow.inference: encoder_proj done");

    const int32_t mel_len1 = prompt_feat.shape()[1];
    const int32_t mel_len_total = h.shape()[1];
    const int32_t mel_len2 = mel_len_total - mel_len1;
    MLLM_RT_ASSERT(mel_len2 > 0);

    auto conds = Tensor::zeros(h.shape(), kFloat32, kCPU);
    // copy prompt mel to prefix.
    auto* cond_ptr = conds.ptr<float>();
    const auto* prm_ptr = prompt_feat.ptr<float>();
    const int32_t C = h.shape()[2];
    for (int32_t t = 0; t < mel_len1; ++t) {
      std::memcpy(cond_ptr + static_cast<int64_t>(t) * C, prm_ptr + static_cast<int64_t>(t) * C, sizeof(float) * C);
    }

    auto feat = decoder_.forwardOne(h.transpose(1, 2).contiguous(), spk, conds.transpose(1, 2).contiguous(), n_timesteps);
    debugLog("flow.inference: decoder done");
    // remove prompt part.
    auto out = feat[{kAll, kAll, {mel_len1, mel_len1 + mel_len2}}].contiguous();
    debugLog("flow.inference: finish");
    return out;
  }

 private:
  MiniCPMO45FlowConfig cfg_;
  nn::Embedding input_embedding_;
  nn::Linear spk_embed_affine_layer_;
  UpsampleConformerEncoderV2 encoder_;
  nn::Linear encoder_proj_;
  CausalConditionalCFM decoder_;
};

class SnakeActivation final : public nn::Module {
 public:
  SnakeActivation() = default;
  SnakeActivation(const std::string& name, int32_t channels) : nn::Module(name) {
    alpha_ = reg<nn::Param>("alpha", getModuleName() + ".alpha", Tensor::shape_t{channels});
  }

  Tensor forwardOne(Tensor x) {
    MLLM_RT_ASSERT_EQ(x.dtype(), kFloat32);
    auto out = Tensor::empty(x.shape(), kFloat32, kCPU).alloc();
    auto in = x.contiguous();
    auto* dst = out.ptr<float>();
    const auto* src = in.ptr<float>();
    const auto* alpha = alpha_.weight().contiguous().ptr<float>();
    const int32_t B = in.shape()[0];
    const int32_t C = in.shape()[1];
    const int32_t T = in.shape()[2];
    const int64_t stride_b = static_cast<int64_t>(C) * T;
    const int64_t stride_c = T;
    constexpr float eps = 1e-9f;
    for (int32_t b = 0; b < B; ++b) {
      for (int32_t c = 0; c < C; ++c) {
        const float a = alpha[c];
        for (int32_t t = 0; t < T; ++t) {
          const int64_t idx = static_cast<int64_t>(b) * stride_b + static_cast<int64_t>(c) * stride_c + t;
          const float v = src[idx];
          const float s = std::sin(v * a);
          dst[idx] = v + (s * s) / (a + eps);
        }
      }
    }
    return out;
  }

 private:
  nn::Param alpha_;
};

class ResBlock final : public nn::Module {
 public:
  ResBlock() = default;
  ResBlock(const std::string& name, int32_t channels, int32_t kernel_size, const std::vector<int32_t>& dilations)
      : nn::Module(name) {
    MLLM_RT_ASSERT_EQ(static_cast<int32_t>(dilations.size()), 3);
    for (int32_t i = 0; i < 3; ++i) {
      convs1_.emplace_back(reg<nn::Conv1D>("convs1." + std::to_string(i), channels, channels, kernel_size, 1,
                                           getPadding(kernel_size, dilations[i]), dilations[i], 1, true));
      convs2_.emplace_back(reg<nn::Conv1D>("convs2." + std::to_string(i), channels, channels, kernel_size, 1,
                                           getPadding(kernel_size, 1), 1, 1, true));
      activations1_.emplace_back(reg<SnakeActivation>("activations1." + std::to_string(i), channels));
      activations2_.emplace_back(reg<SnakeActivation>("activations2." + std::to_string(i), channels));
    }
  }

  Tensor forwardOne(Tensor x) {
    auto out = x;
    for (int32_t i = 0; i < 3; ++i) {
      auto y = activations1_[i].forwardOne(out);
      y = convs1_[i](y);
      y = activations2_[i].forwardOne(y);
      y = convs2_[i](y);
      out = out + y;
    }
    return out;
  }

 private:
  static int32_t getPadding(int32_t kernel_size, int32_t dilation) { return (kernel_size * dilation - dilation) / 2; }

 private:
  std::vector<nn::Conv1D> convs1_;
  std::vector<nn::Conv1D> convs2_;
  std::vector<SnakeActivation> activations1_;
  std::vector<SnakeActivation> activations2_;
};

class ConvRNNF0Predictor final : public nn::Module {
 public:
  ConvRNNF0Predictor() = default;
  ConvRNNF0Predictor(const std::string& name, int32_t in_channels = 80, int32_t cond_channels = 512) : nn::Module(name) {
    condnet_0_ = reg<nn::Conv1D>("condnet.0", in_channels, cond_channels, 3, 1, 1, 1, 1, true);
    condnet_2_ = reg<nn::Conv1D>("condnet.2", cond_channels, cond_channels, 3, 1, 1, 1, 1, true);
    condnet_4_ = reg<nn::Conv1D>("condnet.4", cond_channels, cond_channels, 3, 1, 1, 1, 1, true);
    condnet_6_ = reg<nn::Conv1D>("condnet.6", cond_channels, cond_channels, 3, 1, 1, 1, 1, true);
    condnet_8_ = reg<nn::Conv1D>("condnet.8", cond_channels, cond_channels, 3, 1, 1, 1, 1, true);
    classifier_ = reg<nn::Linear>("classifier", cond_channels, 1, true);
  }

  Tensor forwardOne(Tensor x) {
    auto y = condnet_0_(x);
    y = tensorElu(y);
    y = condnet_2_(y);
    y = tensorElu(y);
    y = condnet_4_(y);
    y = tensorElu(y);
    y = condnet_6_(y);
    y = tensorElu(y);
    y = condnet_8_(y);
    y = tensorElu(y);
    y = y.transpose(1, 2).contiguous();
    y = classifier_(y).squeeze(-1);
    y = tensorAbs(y);
    return y;
  }

 private:
  static Tensor tensorAbs(Tensor x) {
    auto in = x.contiguous();
    auto out = Tensor::empty(in.shape(), kFloat32, kCPU).alloc();
    const auto* src = in.ptr<float>();
    auto* dst = out.ptr<float>();
    const int64_t n = static_cast<int64_t>(in.numel());
    MLLM_CONDITIONAL_PARALLEL_FOR(n > 4096, 4, i, 0, n, 1, { dst[i] = std::abs(src[i]); });
    return out;
  }

  static Tensor tensorElu(Tensor x) {
    auto in = x.contiguous();
    auto out = Tensor::empty(in.shape(), kFloat32, kCPU).alloc();
    const auto* src = in.ptr<float>();
    auto* dst = out.ptr<float>();
    const int64_t n = static_cast<int64_t>(in.numel());
    MLLM_CONDITIONAL_PARALLEL_FOR(n > 4096, 4, i, 0, n, 1, {
      const float v = src[i];
      dst[i] = (v >= 0.0f) ? v : std::expm1(v);
    });
    return out;
  }

 private:
  nn::Conv1D condnet_0_;
  nn::Conv1D condnet_2_;
  nn::Conv1D condnet_4_;
  nn::Conv1D condnet_6_;
  nn::Conv1D condnet_8_;
  nn::Linear classifier_;
};

class SineGen2 {
 public:
  SineGen2() = default;
  SineGen2(int32_t sampling_rate, int32_t upsample_scale, int32_t harmonic_num, float sine_amp, float noise_std, float voiced_threshold)
      : sampling_rate_(sampling_rate),
        upsample_scale_(upsample_scale),
        harmonic_num_(harmonic_num),
        sine_amp_(sine_amp),
        noise_std_(noise_std),
        voiced_threshold_(voiced_threshold) {}

  std::tuple<Tensor, Tensor, Tensor> forward(Tensor f0) {
    // f0: [B, T, 1]
    auto fn = makeHarmonics(f0);
    auto sine = f02sine(fn) * sine_amp_;
    auto uv = f02uv(f0);
    auto inv_uv = uv * -1.0f + 1.0f;
    auto noise_amp = uv * noise_std_ + inv_uv * (sine_amp_ / 3.0f);
    auto noise = randomLike(noise_amp);
    auto out = sine * uv + noise_amp * noise;
    return {out, uv, noise_amp * noise};
  }

 private:
  Tensor makeHarmonics(Tensor f0) const {
    const int32_t B = f0.shape()[0];
    const int32_t T = f0.shape()[1];
    const int32_t H = harmonic_num_ + 1;
    auto out = Tensor::empty({B, T, H}, kFloat32, kCPU).alloc();
    const auto* fp = f0.contiguous().ptr<float>();
    auto* op = out.ptr<float>();
    for (int32_t b = 0; b < B; ++b) {
      for (int32_t t = 0; t < T; ++t) {
        const float v = fp[(static_cast<int64_t>(b) * T + t)];
        for (int32_t h = 0; h < H; ++h) { op[(static_cast<int64_t>(b) * T + t) * H + h] = v * static_cast<float>(h + 1); }
      }
    }
    return out;
  }

  Tensor f02uv(Tensor f0) const {
    auto out = Tensor::empty(f0.shape(), kFloat32, kCPU).alloc();
    const auto* src = f0.contiguous().ptr<float>();
    auto* dst = out.ptr<float>();
    const int64_t n = static_cast<int64_t>(out.numel());
    for (int64_t i = 0; i < n; ++i) { dst[i] = src[i] > voiced_threshold_ ? 1.0f : 0.0f; }
    return out;
  }

  Tensor f02sine(Tensor f0_values) const {
    // f0_values: [B, T, H]
    auto fv = f0_values.contiguous();
    const int32_t B = fv.shape()[0];
    const int32_t T = fv.shape()[1];
    const int32_t H = fv.shape()[2];
    auto rad = Tensor::empty(fv.shape(), kFloat32, kCPU).alloc();
    const auto* fp = fv.ptr<float>();
    auto* rp = rad.ptr<float>();
    for (int32_t b = 0; b < B; ++b) {
      for (int32_t t = 0; t < T; ++t) {
        for (int32_t h = 0; h < H; ++h) {
          const int64_t idx = (static_cast<int64_t>(b) * T + t) * H + h;
          float v = fp[idx] / static_cast<float>(sampling_rate_);
          v = v - std::floor(v);
          rp[idx] = v;
        }
      }
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (int32_t b = 0; b < B; ++b) {
      for (int32_t h = 1; h < H; ++h) {
        const float phase0 = uni(rng);
        rp[(static_cast<int64_t>(b) * T + 0) * H + h] += phase0;
      }
    }

    // linear interpolate in time by 1 / upsample_scale, then cumulative phase, then upsample back.
    auto rad_t = rad.transpose(1, 2).contiguous();                      // [B,H,T]
    auto down_t = nn::functional::interpolateByScale(rad_t, {1.0f / static_cast<float>(upsample_scale_)},
                                                      aops::InterpolateOpMode::kLinear, false, false);
    down_t = down_t.transpose(1, 2).contiguous();                       // [B,T',H]

    auto phase = Tensor::empty(down_t.shape(), kFloat32, kCPU).alloc();
    auto* pp = phase.ptr<float>();
    const auto* dp = down_t.ptr<float>();
    const int32_t Td = down_t.shape()[1];
    for (int32_t b = 0; b < B; ++b) {
      for (int32_t h = 0; h < H; ++h) {
        float acc = 0.0f;
        for (int32_t t = 0; t < Td; ++t) {
          const int64_t idx = (static_cast<int64_t>(b) * Td + t) * H + h;
          acc += dp[idx];
          constexpr float kPi = 3.14159265358979323846f;
          pp[idx] = acc * 2.0f * kPi;
        }
      }
    }

    auto phase_t = phase.transpose(1, 2).contiguous();                  // [B,H,T']
    phase_t = phase_t * static_cast<float>(upsample_scale_);
    auto up_t = nn::functional::interpolateByScale(phase_t, {static_cast<float>(upsample_scale_)},
                                                    aops::InterpolateOpMode::kLinear, false, false);
    up_t = up_t.transpose(1, 2).contiguous();                           // [B,T,H]
    auto out = nn::functional::sin(up_t);
    return out;
  }

  static Tensor randomLike(Tensor x) {
    auto out = Tensor::empty(x.shape(), kFloat32, kCPU).alloc();
    auto* dst = out.ptr<float>();
    const int64_t n = static_cast<int64_t>(out.numel());
    static thread_local std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) { dst[i] = dist(rng); }
    return out;
  }

 private:
  int32_t sampling_rate_ = 24000;
  int32_t upsample_scale_ = 480;
  int32_t harmonic_num_ = 8;
  float sine_amp_ = 0.1f;
  float noise_std_ = 0.003f;
  float voiced_threshold_ = 10.0f;
};

class SourceModuleHnNSF2 {
 public:
  SourceModuleHnNSF2() = default;
  SourceModuleHnNSF2(int32_t sampling_rate, int32_t upsample_scale, int32_t harmonic_num, float sine_amp, float noise_std,
                     float voiced_threshold)
      : l_sin_gen_(sampling_rate, upsample_scale, harmonic_num, sine_amp, noise_std, voiced_threshold),
        sine_amp_(sine_amp) {}

  // This wrapper only supports loading external weight via setLinearWeights().
  void setLinearWeights(Tensor w, Tensor b) {
    linear_w_ = w.contiguous();
    linear_b_ = b.contiguous();
  }

  std::tuple<Tensor, Tensor, Tensor> forward(Tensor x) {
    auto [sine_wavs, uv, _] = l_sin_gen_.forward(x);
    auto sine_merge = linearForward(sine_wavs);
    sine_merge = tensorTanh(sine_merge);
    auto noise = randomLike(uv) * (sine_amp_ / 3.0f);
    return {sine_merge, noise, uv};
  }

 private:
  Tensor linearForward(Tensor x) {
    // x: [B,T,H], weight [1,H]
    MLLM_RT_ASSERT(!linear_w_.isNil());
    MLLM_RT_ASSERT(!linear_b_.isNil());
    const int32_t B = x.shape()[0];
    const int32_t T = x.shape()[1];
    const int32_t H = x.shape()[2];
    auto out = Tensor::empty({B, T, 1}, kFloat32, kCPU).alloc();
    const auto* xp = x.contiguous().ptr<float>();
    const auto* wp = linear_w_.contiguous().ptr<float>();
    const float bias = linear_b_.constAt<float>({0});
    auto* op = out.ptr<float>();
    for (int32_t b = 0; b < B; ++b) {
      for (int32_t t = 0; t < T; ++t) {
        float acc = bias;
        for (int32_t h = 0; h < H; ++h) { acc += xp[(static_cast<int64_t>(b) * T + t) * H + h] * wp[h]; }
        op[static_cast<int64_t>(b) * T + t] = acc;
      }
    }
    return out;
  }

  static Tensor randomLike(Tensor x) {
    auto out = Tensor::empty(x.shape(), kFloat32, kCPU).alloc();
    auto* dst = out.ptr<float>();
    const int64_t n = static_cast<int64_t>(out.numel());
    static thread_local std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) { dst[i] = dist(rng); }
    return out;
  }

  static Tensor tensorTanh(Tensor x) {
    auto in = x.contiguous();
    auto out = Tensor::empty(in.shape(), kFloat32, kCPU).alloc();
    const auto* src = in.ptr<float>();
    auto* dst = out.ptr<float>();
    const int64_t n = static_cast<int64_t>(in.numel());
    for (int64_t i = 0; i < n; ++i) { dst[i] = std::tanh(src[i]); }
    return out;
  }

 private:
  SineGen2 l_sin_gen_;
  float sine_amp_ = 0.1f;
  Tensor linear_w_ = Tensor::nil();
  Tensor linear_b_ = Tensor::nil();
};

class HiFTGenerator final : public nn::Module {
 public:
  HiFTGenerator() = default;
  HiFTGenerator(const std::string& name, const MiniCPMO45HiFTConfig& cfg)
      : nn::Module(name), cfg_(cfg),
        upsample_total_scale_(cfg.upsample_rates[0] * cfg.upsample_rates[1] * cfg.upsample_rates[2] * cfg.istft_hop_len),
        m_source_(cfg.sampling_rate, upsample_total_scale_, cfg.nb_harmonics, cfg.nsf_alpha, cfg.nsf_sigma, cfg.nsf_voiced_threshold) {
    conv_pre_ = reg<nn::Conv1D>("conv_pre", cfg.in_channels, cfg.base_channels, 7, 1, 3, 1, 1, true);
    for (int32_t i = 0; i < static_cast<int32_t>(cfg.upsample_rates.size()); ++i) {
      const int32_t in_ch = cfg.base_channels / static_cast<int32_t>(std::pow(2, i));
      const int32_t out_ch = cfg.base_channels / static_cast<int32_t>(std::pow(2, i + 1));
      ups_.emplace_back(
          reg<nn::ConvTranspose1D>("ups." + std::to_string(i), in_ch, out_ch, cfg.upsample_kernel_sizes[i], cfg.upsample_rates[i],
                                   (cfg.upsample_kernel_sizes[i] - cfg.upsample_rates[i]) / 2, 0, 1, 1, true));
    }

    // source downs
    std::vector<int32_t> downsample_rates = {1, cfg.upsample_rates[2], cfg.upsample_rates[2] * cfg.upsample_rates[1]};
    std::reverse(downsample_rates.begin(), downsample_rates.end());  // [15,3,1]
    for (int32_t i = 0; i < static_cast<int32_t>(downsample_rates.size()); ++i) {
      const int32_t u = downsample_rates[i];
      const int32_t out_ch = cfg.base_channels / static_cast<int32_t>(std::pow(2, i + 1));
      if (u == 1) {
        source_downs_.emplace_back(reg<nn::Conv1D>("source_downs." + std::to_string(i), cfg.istft_n_fft + 2, out_ch, 1, 1, 0, 1, 1, true));
      } else {
        source_downs_.emplace_back(reg<nn::Conv1D>("source_downs." + std::to_string(i), cfg.istft_n_fft + 2, out_ch, u * 2, u, (u / 2), 1, 1, true));
      }
      source_resblocks_.emplace_back(
          reg<ResBlock>("source_resblocks." + std::to_string(i), out_ch, cfg.source_resblock_kernel_sizes[i], cfg.source_resblock_dilation_sizes[i]));
    }

    const int32_t num_ups = static_cast<int32_t>(cfg.upsample_rates.size());
    const int32_t num_kernels = static_cast<int32_t>(cfg.resblock_kernel_sizes.size());
    for (int32_t i = 0; i < num_ups; ++i) {
      const int32_t ch = cfg.base_channels / static_cast<int32_t>(std::pow(2, i + 1));
      for (int32_t j = 0; j < num_kernels; ++j) {
        resblocks_.emplace_back(reg<ResBlock>("resblocks." + std::to_string(static_cast<int32_t>(resblocks_.size())), ch,
                                              cfg.resblock_kernel_sizes[j], cfg.resblock_dilation_sizes[j]));
      }
    }

    conv_post_ = reg<nn::Conv1D>("conv_post", cfg.base_channels / static_cast<int32_t>(std::pow(2, cfg.upsample_rates.size())),
                                 cfg.istft_n_fft + 2, 7, 1, 3, 1, 1, true);
    f0_predictor_ = reg<ConvRNNF0Predictor>("f0_predictor");
    stft_ = reg<nn::STFT>("internal_stft", cfg.istft_n_fft, cfg.istft_hop_len, cfg.istft_n_fft, true, true, "reflect", false);
    istft_ = reg<nn::ISTFT>("internal_istft", cfg.istft_n_fft, cfg.istft_hop_len, cfg.istft_n_fft, true, true, "reflect");
    hann_window_ = makeHannWindow(cfg.istft_n_fft);
  }

  void loadFromParameter(const ParameterFile::ptr_t& param) {
    nn::Module::load(param);
    // SourceModuleHnNSF2 linear is not a nn::Module member, load manually.
    auto w = param->pull(getModuleName() + ".m_source.l_linear.weight");
    auto b = param->pull(getModuleName() + ".m_source.l_linear.bias");
    if (w.dtype() != kFloat32) { w = w.to(kFloat32); }
    if (b.dtype() != kFloat32) { b = b.to(kFloat32); }
    w = w.contiguous().view({1, cfg_.nb_harmonics + 1});
    b = b.contiguous().view({1});
    m_source_.setLinearWeights(w, b);
  }

  Tensor forwardOne(Tensor speech_feat) {
    auto f0 = f0_predictor_.forwardOne(speech_feat);                             // [B,T]
    auto f0_ex = f0.view({f0.shape()[0], 1, f0.shape()[1]});                     // [B,1,T]
    auto s = repeatInterleave1d(f0_ex, upsample_total_scale_).transpose(1, 2);   // [B,S,1]
    auto [s_merge, _, _uv] = m_source_.forward(s);
    auto src = s_merge.transpose(1, 2).contiguous();                              // [B,1,S]
    auto wav = decode(speech_feat, src);
    return wav;
  }

 private:
  Tensor decode(Tensor x_in, Tensor s) {
    auto stft = stft_(s.squeeze(1), hann_window_);  // [B,F,T,2]
    auto stft_chunks = nn::functional::chunk<2>(stft, 3);
    auto s_real = stft_chunks[0].squeeze(-1);
    auto s_imag = stft_chunks[1].squeeze(-1);
    auto s_stft = nn::functional::concat({s_real, s_imag}, 1);                   // [B,F*2,T]

    auto x = conv_pre_(x_in);
    const int32_t num_ups = static_cast<int32_t>(ups_.size());
    const int32_t num_kernels = static_cast<int32_t>(cfg_.resblock_kernel_sizes.size());
    for (int32_t i = 0; i < num_ups; ++i) {
      x = tensorLeakyRelu(x, cfg_.lrelu_slope);
      x = ups_[i](x);
      if (i == num_ups - 1) { x = nn::functional::pad(x, {1, 0}, aops::PadMode::kReflect); }

      auto si = source_downs_[i](s_stft);
      si = source_resblocks_[i].forwardOne(si);
      x = x + si;

      Tensor xs = Tensor::nil();
      for (int32_t j = 0; j < num_kernels; ++j) {
        auto y = resblocks_[i * num_kernels + j].forwardOne(x);
        if (j == 0) {
          xs = y;
        } else {
          xs = xs + y;
        }
      }
      x = xs / static_cast<float>(num_kernels);
    }

    x = tensorLeakyRelu(x, 0.01f);
    x = conv_post_(x);                                                           // [B,18,T]
    auto mag = x[{kAll, {0, cfg_.istft_n_fft / 2 + 1}, kAll}].contiguous();
    auto phase = x[{kAll, {cfg_.istft_n_fft / 2 + 1, cfg_.istft_n_fft + 2}, kAll}].contiguous();
    mag = nn::functional::exp(mag);
    mag = nn::functional::clip(mag, 0.0f, 1e2f);
    // Keep parity with python HiFT: phase is first squashed by sin() before ISTFT synthesis.
    phase = nn::functional::sin(phase);
    auto real = mag * nn::functional::cos(phase);
    auto imag = mag * nn::functional::sin(phase);
    auto S = real + std::complex<float>{0, 1} * imag;
    auto wav = istft_(S, hann_window_);
    wav = nn::functional::clip(wav, -cfg_.audio_limit, cfg_.audio_limit);
    return wav;
  }

 private:
  MiniCPMO45HiFTConfig cfg_;
  int32_t upsample_total_scale_ = 480;
  nn::Conv1D conv_pre_;
  std::vector<nn::ConvTranspose1D> ups_;
  std::vector<nn::Conv1D> source_downs_;
  std::vector<ResBlock> source_resblocks_;
  std::vector<ResBlock> resblocks_;
  nn::Conv1D conv_post_;
  ConvRNNF0Predictor f0_predictor_;
  nn::STFT stft_;
  nn::ISTFT istft_;
  Tensor hann_window_ = Tensor::nil();
  SourceModuleHnNSF2 m_source_;
};

class MiniCPMO45Token2WavModel final : public nn::Module {
 public:
  MiniCPMO45Token2WavModel() = default;
  MiniCPMO45Token2WavModel(const std::string& name, const MiniCPMO45Token2WavConfig& cfg) : nn::Module(name), cfg_(cfg) {
    flow_model_ = reg<CausalMaskedDiffWithXvec>("flow_model", cfg_.flow);
    hift_model_ = reg<HiFTGenerator>("hift_model", cfg_.hift);
  }

  void loadFromParameter(const ParameterFile::ptr_t& param_file) {
    // Materialize weight_norm reparameterized conv weights in-place.
    (void)materializeWeightNormParameters(param_file, getModuleName() + ".hift_model.");
    flow_model_.load(param_file);
    hift_model_.loadFromParameter(param_file);
  }

  Tensor infer(const std::vector<int64_t>& token_ids, const MiniCPMO45Token2WavPromptCache& prompt_cache, int32_t n_timesteps) {
    if (token_ids.empty()) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "MiniCPM-o-4_5 token2wav got empty token ids."); }
    if (prompt_cache.prompt_tokens.empty() || prompt_cache.prompt_mels.isNil() || prompt_cache.spk_emb.isNil()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "MiniCPM-o-4_5 token2wav prompt cache is incomplete.");
    }

    auto token = Tensor::empty({1, static_cast<int32_t>(token_ids.size())}, kInt64, kCPU).alloc();
    for (int32_t i = 0; i < static_cast<int32_t>(token_ids.size()); ++i) { token.at<int64_t>({0, i}) = token_ids[static_cast<size_t>(i)]; }

    auto prompt_token = Tensor::empty({1, static_cast<int32_t>(prompt_cache.prompt_tokens.size())}, kInt64, kCPU).alloc();
    for (int32_t i = 0; i < static_cast<int32_t>(prompt_cache.prompt_tokens.size()); ++i) {
      prompt_token.at<int64_t>({0, i}) = static_cast<int64_t>(prompt_cache.prompt_tokens[static_cast<size_t>(i)]);
    }

    auto prompt_mels = Tensor(prompt_cache.prompt_mels);
    auto spk = Tensor(prompt_cache.spk_emb);
    if (prompt_mels.dtype() != kFloat32) { prompt_mels = prompt_mels.to(kFloat32); }
    if (spk.dtype() != kFloat32) { spk = spk.to(kFloat32); }

    auto mel = flow_model_.inference(token, prompt_token, prompt_mels, spk, n_timesteps);
    auto wav = hift_model_.forwardOne(mel);
    return wav;
  }

 private:
  MiniCPMO45Token2WavConfig cfg_;
  CausalMaskedDiffWithXvec flow_model_;
  HiFTGenerator hift_model_;
};

}  // namespace token2wav

using token2wav::MiniCPMO45Token2WavModel;

}  // namespace mllm::models::minicpm_o45
