// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <unordered_map>

#include "KernelTestHelper.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"

using namespace mllm;  // NOLINT

void naive_conv_transpose1d(const float* input_data, const float* weight_data, const float* bias_data, float* output_data,
                            int batch, int in_channels, int sequence, int out_channels, int kernel_size, int stride,
                            int padding, int dilation, int output_padding, int groups) {
  const int out_sequence = (sequence - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
  std::fill_n(output_data, batch * out_channels * out_sequence, 0.0f);

  const int in_channels_per_group = in_channels / groups;
  const int out_channels_per_group = out_channels / groups;

  for (int b = 0; b < batch; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      const int group_idx = oc / out_channels_per_group;
      const int oc_in_group = oc % out_channels_per_group;
      for (int out_pos = 0; out_pos < out_sequence; ++out_pos) {
        float sum = 0.0f;
        for (int ic_in_group = 0; ic_in_group < in_channels_per_group; ++ic_in_group) {
          const int ic = group_idx * in_channels_per_group + ic_in_group;
          const int base_input_idx = b * (in_channels * sequence) + ic * sequence;
          const int base_weight_idx = (ic * out_channels_per_group + oc_in_group) * kernel_size;

          for (int k = 0; k < kernel_size; ++k) {
            int input_pos = out_pos + padding - k * dilation;
            if (input_pos % stride != 0) { continue; }
            input_pos /= stride;
            if (input_pos < 0 || input_pos >= sequence) { continue; }

            const int input_idx = base_input_idx + input_pos;
            const int weight_idx = base_weight_idx + k;
            sum += input_data[input_idx] * weight_data[weight_idx];
          }
        }
        if (bias_data != nullptr) { sum += bias_data[oc]; }
        const int output_idx = b * (out_channels * out_sequence) + oc * out_sequence + out_pos;
        output_data[output_idx] = sum;
      }
    }
  }
}

class ConvTranspose1DModule : public nn::Module {
  nn::ConvTranspose1D conv_;

 public:
  ConvTranspose1DModule(int in_channel, int out_channel, int kernel_size, int stride, int padding, int output_padding,
                        int dilation, int groups, bool bias) {
    conv_ = reg<nn::ConvTranspose1D>("conv", in_channel, out_channel, kernel_size, stride, padding, output_padding, dilation,
                                    groups, bias);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {conv_(inputs[0])};
  }
};

class ConvTranspose1DKernelTest : public KernelTest {
 public:
  bool testConvTranspose1DOnce(const std::unordered_map<std::string, int32_t>& cfg) {
    auto batch = cfg.at("batch");
    auto in_channel = cfg.at("in_channel");
    auto out_channel = cfg.at("out_channel");
    auto sequence = cfg.at("sequence");
    auto kernel_size = cfg.at("kernel_size");
    auto stride = cfg.at("stride");
    auto padding = cfg.at("padding");
    auto output_padding = cfg.at("output_padding");
    auto dilation = cfg.at("dilation");
    auto groups = cfg.at("groups");
    auto bias = cfg.at("bias");

    auto module = ConvTranspose1DModule(in_channel, out_channel, kernel_size, stride, padding, output_padding, dilation,
                                        groups, bias);

    auto weight_param =
        Tensor::random({in_channel, out_channel / groups, kernel_size}, -1, 1, kFloat32, kCPU);
    auto bias_param = Tensor::random({out_channel}, -1, 1, kFloat32, kCPU);
    weight_param.setName("conv.weight");
    bias_param.setName("conv.bias");

    auto param = ParameterFile::create();
    param->push("conv.weight", weight_param);
    if (bias) { param->push("conv.bias", bias_param); }
    module.load(param);

    auto input = Tensor::random({batch, in_channel, sequence}, -1, 1, kFloat32, kCPU);
    auto predict = module(input)[0];

    auto expected = Tensor::zeros(predict.shape(), kFloat32, kCPU);
    naive_conv_transpose1d(input.ptr<float>(), weight_param.ptr<float>(), bias ? bias_param.ptr<float>() : nullptr,
                           expected.ptr<float>(), batch, in_channel, sequence, out_channel, kernel_size, stride, padding,
                           dilation, output_padding, groups);

    auto result = test::allClose(expected, predict, 1e-4f, 1e-4f);
    if (!result) {
      print(result);
      return false;
    }
    return true;
  }

  bool testConvTranspose1D(const std::vector<std::unordered_map<std::string, int32_t>>& cfgs) {
    for (auto& cfg : cfgs) {
      if (!testConvTranspose1DOnce(cfg)) {
        auto batch = cfg.at("batch");
        auto in_channel = cfg.at("in_channel");
        auto out_channel = cfg.at("out_channel");
        auto sequence = cfg.at("sequence");
        auto kernel_size = cfg.at("kernel_size");
        auto stride = cfg.at("stride");
        auto padding = cfg.at("padding");
        auto output_padding = cfg.at("output_padding");
        auto dilation = cfg.at("dilation");
        auto groups = cfg.at("groups");
        auto bias = cfg.at("bias");
        print(batch, in_channel, out_channel, sequence, kernel_size, stride, padding, output_padding, dilation, groups, bias);
        return false;
      }
    }
    return true;
  }
};
