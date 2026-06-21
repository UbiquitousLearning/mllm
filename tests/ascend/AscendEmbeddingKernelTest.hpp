// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Functional.hpp"
#include "KernelTestHelper.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

class AscendEmbeddingKernelTest : public KernelTest {
 public:
  AscendEmbeddingKernelTest() = default;
  ~AscendEmbeddingKernelTest() override = default;

  // Test Gather operation using nn::functional::gather
  // test_cases: {batch_size, seq_len, vocab_size, hidden_size}
  bool EmbeddingFloat16Test(const std::vector<std::tuple<int, int, int, int>>& test_cases) {
    using namespace mllm;  // NOLINT
    for (auto& test_case : test_cases) {
      int batch_size = std::get<0>(test_case);
      int seq_len = std::get<1>(test_case);
      int vocab_size = std::get<2>(test_case);
      int hidden_size = std::get<3>(test_case);

      std::cout << "[GatherTest] Testing B=" << batch_size << ", S=" << seq_len
                << ", vocab=" << vocab_size << ", hidden=" << hidden_size << std::endl;

      // 1. Create random weight table on CPU (FP16)
      // weight: [vocab_size, hidden_size]
      Tensor weight_cpu = Tensor::random({vocab_size, hidden_size}, -1.0f, 1.0f, kFloat16, kCPU);

      // 2. Create random indices on CPU (INT32)
      // indices: [batch_size, seq_len]
      Tensor indices_cpu = Tensor::zeros({batch_size, seq_len}, kInt32, kCPU);
      {
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, vocab_size - 1);
        auto* idx_ptr = indices_cpu.ptr<int32_t>();
        for (int i = 0; i < batch_size * seq_len; ++i) {
          idx_ptr[i] = dist(gen);
        }
      }

      // 3. Compute reference result on CPU
      // output: [batch_size, seq_len, hidden_size]
      Tensor ref_cpu = Tensor::zeros({batch_size, seq_len, hidden_size}, kFloat16, kCPU);
      {
        auto* weight_ptr = weight_cpu.ptr<mllm_fp16_t>();
        auto* idx_ptr = indices_cpu.ptr<int32_t>();
        auto* out_ptr = ref_cpu.ptr<mllm_fp16_t>();

        for (int b = 0; b < batch_size; ++b) {
          for (int s = 0; s < seq_len; ++s) {
            int token_idx = idx_ptr[b * seq_len + s];
            for (int h = 0; h < hidden_size; ++h) {
              out_ptr[(b * seq_len + s) * hidden_size + h] = weight_ptr[token_idx * hidden_size + h];
            }
          }
        }
      }

      // 4. Move weight to Ascend
      auto weight_ascend = weight_cpu.to(kAscend);

      // 5. Move indices to Ascend
      auto indices_ascend = indices_cpu.to(kAscend);

      // 6. Run gather on Ascend (dim=0 for embedding-like behavior)
      auto y_ascend = nn::functional::gather(weight_ascend, 0, indices_ascend);

      // 7. Move result back to CPU and compare with reference
      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-5f, 1e-5f);
      if (!result.is_close) {
        std::cout << "[GatherTest] FAILED!" << std::endl;
        return false;
      }
      std::cout << "[GatherTest] PASSED" << std::endl;
    }
    return true;
  }
};
