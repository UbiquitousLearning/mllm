// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>

#include "KernelTestHelper.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"

class TanhModule : public mllm::nn::Module {
  mllm::nn::Tanh tanh_;

 public:
  TanhModule() { tanh_ = reg<mllm::nn::Tanh>("tanh"); }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& args) override {
    return {tanh_(inputs[0])};
  }
};

class TanhKernelTest : public KernelTest {
 public:
  bool testTanh(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using mllm::Tensor;
    using mllm::kCPU;
    using mllm::kFloat32;
    TanhModule module;

    for (auto& s : shapes) {
      auto input = Tensor::random(s, -3, 3, kFloat32, kCPU);
      auto output = module(input)[0];
      auto expected = Tensor::empty(s, kFloat32, kCPU).alloc();

      const auto* in_ptr = input.ptr<mllm::mllm_fp32_t>();
      auto* out_ptr = expected.ptr<mllm::mllm_fp32_t>();
      const auto numel = input.numel();
      for (size_t i = 0; i < numel; ++i) { out_ptr[i] = std::tanh(in_ptr[i]); }

      auto result = mllm::test::allClose(expected, output, 1e-5f, 1e-5f);
      if (!result) {
        mllm::print(result);
        return false;
      }
    }
    return true;
  }
};
