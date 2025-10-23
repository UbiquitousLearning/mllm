// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <unordered_map>

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/core/ParameterFile.hpp"

#include "KernelTestHelper.hpp"

using namespace mllm;  // NOLINT

void naive_conv2d(const float* input_data, const float* weight_data, const float* bias_data, float* output_data,
                  int in_channels, int in_h, int in_w, int out_channels, int kernel_h, int kernel_w, int pad_h, int pad_w,
                  int stride_h, int stride_w, int dilation_h, int dilation_w) {
  const int out_h = (in_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int out_w = (in_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  for (int oc = 0; oc < out_channels; ++oc) {
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        float accumulated_value = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
          for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
              const int iy = oy * stride_h + ky * dilation_h - pad_h;
              const int ix = ox * stride_w + kx * dilation_w - pad_w;

              if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                int input_idx = ic * (in_h * in_w) + iy * in_w + ix;
                int weight_idx = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + ky * kernel_w + kx;

                accumulated_value += input_data[input_idx] * weight_data[weight_idx];
              }
            }
          }
        }

        if (bias_data != nullptr) { accumulated_value += bias_data[oc]; }

        int output_idx = oc * (out_h * out_w) + oy * out_w + ox;
        output_data[output_idx] = accumulated_value;
      }
    }
  }
}

class Conv2DModule : public nn::Module {
  nn::Conv2D conv2d_;

 public:
  Conv2DModule() = default;

  Conv2DModule(int in_channel, int out_channel, int K_H, int K_W, int S_H, int S_W, int P_H, int P_W, bool bias)
      : nn::Module() {
    conv2d_ = reg<nn::Conv2D>("emb", in_channel, out_channel, std::vector<int>{K_H, K_W}, std::vector<int>{S_H, S_W},
                              std::vector<int>{P_H, P_W}, std::vector<int>{1, 1}, bias);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // inputs is Q, K_indices, V_indices
    return {conv2d_(inputs[0])};
  }
};

class Conv2DKernelTest : public KernelTest {
 public:
  Conv2DKernelTest() = default;
  ~Conv2DKernelTest() override = default;

  bool testConv2DOnce(const std::unordered_map<std::string, int32_t>& cfg) {
    auto in_channel = cfg.at("in_channel");
    auto out_channel = cfg.at("out_channel");
    auto I_H = cfg.at("I_H");
    auto I_W = cfg.at("I_W");
    auto K_H = cfg.at("K_H");
    auto K_W = cfg.at("K_W");
    auto S_H = cfg.at("S_H");
    auto S_W = cfg.at("S_W");
    auto P_H = cfg.at("P_H");
    auto P_W = cfg.at("P_W");
    auto bias = cfg.at("bias");

    auto module = Conv2DModule(in_channel, out_channel, K_H, K_W, S_H, S_W, P_H, P_W, bias);

    // Make fake data
    auto weight_param = Tensor::random({out_channel, in_channel, K_H, K_W}, -1, 1, kFloat32, kCPU);
    auto bias_param = Tensor::random({out_channel}, -1, 1, kFloat32, kCPU);
    weight_param.setName("emb.weight");
    bias_param.setName("emb.bias");
    auto param = ParameterFile::create();
    param->push("emb.weight", weight_param);
    param->push("emb.bias", bias_param);
    module.load(param);

    auto input = Tensor::random({1, in_channel, I_H, I_W}, -1, 1, kFloat32, kCPU);
    auto predict = module(input)[0];

    auto GT = Tensor::empty(predict.shape(), kFloat32, kCPU).alloc();

    // Naive impl to check correctness.
    naive_conv2d(input.ptr<float>(), weight_param.ptr<float>(), bias ? bias_param.ptr<float>() : nullptr, GT.ptr<float>(),
                 in_channel, I_H, I_W, out_channel, K_H, K_W, P_H, P_W, S_H, S_W, 1, 1);

    auto result = test::allClose(GT, predict, 1e-2f, 1e-2f);
    if (!result) {
      print(result);
      return false;
    }

    return true;
  }

  bool testConv2D(const std::vector<std::unordered_map<std::string, int32_t>>& cfgs) {
    for (auto& cfg : cfgs) {
      if (!testConv2DOnce(cfg)) {
        auto in_channel = cfg.at("in_channel");
        auto out_channel = cfg.at("out_channel");
        auto I_H = cfg.at("I_H");
        auto I_W = cfg.at("I_W");
        auto K_H = cfg.at("K_H");
        auto K_W = cfg.at("K_W");
        auto S_H = cfg.at("S_H");
        auto S_W = cfg.at("S_W");
        auto P_H = cfg.at("P_H");
        auto P_W = cfg.at("P_W");
        auto bias = cfg.at("bias");
        print(in_channel, out_channel, I_H, I_W, K_H, K_W, S_H, S_W, P_H, P_W, bias);
        return false;
      }
    }
    return true;
  }
};
