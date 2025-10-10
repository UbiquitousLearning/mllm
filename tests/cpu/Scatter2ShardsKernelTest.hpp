// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"

class Scatter2ShardsKernelTest : public KernelTest {
 public:
  Scatter2ShardsKernelTest() = default;
  ~Scatter2ShardsKernelTest() override = default;

  bool testScatter2Shards() {
    // B, S, H, D
    auto Q = mllm::Tensor::random({1, 100, 8, 64}, -10, 10);
    auto P = mllm::Tensor::zeros({1, 100, 8, 64});

    std::vector<char*> indices;
    indices.reserve(100);
    for (int i = 0; i < 100; ++i) { indices.push_back((char*)P.offsettedPtr<float>({0, i, 0, 0})); }

    auto tensor_indices = mllm::Tensor::refVectorData(indices, {100}, mllm::kInt64);

    mllm::nn::functional::scatter2Shards(Q, tensor_indices, 1);

    return mllm::test::allClose(Q, P).is_close;
  }
};
