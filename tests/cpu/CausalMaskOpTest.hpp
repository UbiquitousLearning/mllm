#pragma once

#include <algorithm>

#include "KernelTestHelper.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/layers/CausalMask.hpp"

class CausalMaskOpTest : public KernelTest {
 public:
  void SetUp() override {
    KernelTest::SetUp();
    mask_.to(mllm::kCPU);
  }

  mllm::test::AllCloseResult runScenario(int B, int H, int S, int D) {
    using namespace mllm;  // NOLINT
    const int64_t total = static_cast<int64_t>(B) * H * S * D;
    auto input = Tensor::arange(0, static_cast<float>(total), 1, kFloat32, kCPU).view({B, H, S, D});
    auto output = mask_(input);
    auto expected = buildExpectedTensor(input);
    auto result = test::allClose(expected, output);
    if (!result) {
      mllm::print(result);
      mllm::print(expected);
      mllm::print(output);
    }
    return result;
  }

 private:
  static mllm::Tensor buildExpectedTensor(const mllm::Tensor& input) {
    using namespace mllm;  // NOLINT
    auto shape = input.shape();
    const int B = shape[0];
    const int H = shape[1];
    const int S = shape[2];
    const int D = shape[3];
    auto expected = Tensor::zeros(shape, kFloat32, kCPU);

    const float* in_ptr = input.ptr<float>();
    float* exp_ptr = expected.ptr<float>();
    const int context_offset = std::max(0, D - S);
    const float mask_value = -1e10f;

    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        for (int s = 0; s < S; ++s) {
          const int allowed = std::min(D, context_offset + s + 1);
          for (int d = 0; d < D; ++d) {
            const int64_t idx = (((static_cast<int64_t>(b) * H) + h) * S + s) * D + d;
            if (d < allowed) {
              exp_ptr[idx] = in_ptr[idx];
            } else {
              exp_ptr[idx] = mask_value;
            }
          }
        }
      }
    }
    return expected;
  }

  mllm::nn::CausalMask mask_;
};

