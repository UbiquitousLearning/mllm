// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/mllm.hpp"
#include "KernelTestHelper.hpp"
#include "mllm/nn/Functional.hpp"

class FlashAttention2KernelTest : public KernelTest {
 public:
  FlashAttention2KernelTest() = default;
  ~FlashAttention2KernelTest() override = default;

  bool oneCase(const std::pair<mllm::Tensor::shape_t, mllm::Tensor::shape_t>& shape) {
    auto q_shape = shape.first;
    auto kv_shape = shape.second;
    auto B = q_shape[0];
    auto Query_S = q_shape[1];
    auto Query_Head = q_shape[2];
    auto Key_S = kv_shape[1];
    auto Key_Head = kv_shape[2];
    auto D = q_shape[3];

    // [B, S, H, D]
    auto Q = mllm::Tensor::random(q_shape, mllm::kFloat16, mllm::kCPU);
    auto K = mllm::Tensor::random(kv_shape, mllm::kFloat16, mllm::kCPU);
    auto V = mllm::Tensor::random(kv_shape, mllm::kFloat16, mllm::kCPU);

    auto output = mllm::nn::functional::flashAttention2(Q, K, V);

    auto ref_output = mllm::Tensor::nil();

    {
      auto weight = mllm::nn::functional::matmul(Q, K, false, true);
      // TODO mask weight
      weight = mllm::nn::functional::softmax(weight, -1);
      ref_output = mllm::nn::functional::matmul(weight, V);
    }

    auto result = mllm::test::allClose(output, ref_output);
    return result.is_close;
  }

  bool manyCases(const std::vector<std::pair<mllm::Tensor::shape_t, mllm::Tensor::shape_t>>& shapes) {
    for (const auto& s : shapes) {
      if (!oneCase(s)) { return false; }
    }
    return true;
  }
};