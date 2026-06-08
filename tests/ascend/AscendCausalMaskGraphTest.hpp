// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "KernelTestHelper.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/graph/AscendCausalMaskPluginOperation.hpp"
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"
#include "mllm/backends/ascend/graph/AscendGraphExecutor.hpp"
#include "mllm/core/Tensor.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

class AscendCausalMaskGraphTest : public KernelTest {
 public:
  bool CausalMaskPluginGraphAccuracyTest() {
    return runScenario("append", 1, 2, 4, 6);
  }

  bool CausalMaskPluginGraphPrefillTest() {
    return runScenario("prefill", 1, 4, 8, 8);
  }

  bool CausalMaskPluginGraphDecodeTest() {
    return runScenario("decode", 1, 4, 1, 8);
  }

 private:
  bool runScenario(const std::string& scenario_name, int B, int H, int S, int D) {
    using namespace mllm;

    std::cout << "[CausalMaskGraphTest] Testing CausalMask plugin graph scenario=" << scenario_name
              << " shape=[" << B << ", " << H << ", " << S << ", " << D << "]" << std::endl;

    std::vector<float> host_input(static_cast<size_t>(B * H * S * D));
    for (size_t i = 0; i < host_input.size(); ++i) {
      host_input[i] = (static_cast<int>(i % 13) - 6) * 0.25f;
    }

    std::vector<mllm_fp16_t> host_input_fp16(host_input.size());
    for (size_t i = 0; i < host_input.size(); ++i) {
      host_input_fp16[i] = static_cast<mllm_fp16_t>(host_input[i]);
    }

    auto input_cpu = Tensor::fromVector<mllm_fp16_t>(host_input_fp16, {B, H, S, D}, kFloat16, kCPU);
    auto input = input_cpu.to(kAscend);
    auto output = Tensor::empty({B, H, S, D}, kFloat16, kAscend).alloc();

    auto inferShape = [](const atb::SVector<atb::TensorDesc>& inTensorDescs,
                         atb::SVector<atb::TensorDesc>& outTensorDescs) -> atb::Status {
      if (inTensorDescs.size() != 1 || outTensorDescs.size() != 1) {
        return atb::ERROR_INVALID_TENSOR_NUM;
      }
      outTensorDescs.at(0) = inTensorDescs.at(0);
      return atb::NO_ERROR;
    };

    ascend::AscendGraphBuilder builder;
    builder.beginGraph("CausalMaskPluginGraph", {"input"}, {"output"}, inferShape);
    builder.addOperation(mllm::ascend::createCausalMaskPluginGraphOp(), {"input"}, {"output"});

    atb::Operation* graph_op = builder.build();
    if (graph_op == nullptr) {
      std::cerr << "[CausalMaskGraphTest] Failed to build graph for scenario=" << scenario_name << std::endl;
      return false;
    }

    ascend::AscendGraphExecutor executor(graph_op, ascend::getGlobalAtbContext());
    std::vector<Tensor> inputs = {input};
    std::vector<Tensor> outputs = {output};
    executor.execute(inputs, outputs);

    auto actual = ascend::copyAscendTensorToHost(output);
    std::vector<float> expected = host_input;
    constexpr float mask_val = -65500.0f;
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        for (int s = 0; s < S; ++s) {
          const int fill_start = D - S + s + 1;
          for (int d = fill_start; d < D; ++d) {
            const size_t idx = static_cast<size_t>(((b * H + h) * S + s) * D + d);
            expected[idx] = mask_val;
          }
        }
      }
    }

    const float atol = 1e-2f;
    const float rtol = 1e-2f;
    size_t mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      const float diff = std::abs(actual[i] - expected[i]);
      const float thr = atol + rtol * std::abs(expected[i]);
      if (diff > thr) {
        if (mismatches < 5) {
          std::cerr << "[CausalMaskGraphTest][" << scenario_name << "] mismatch idx=" << i
                    << ", actual=" << actual[i]
                    << ", expected=" << expected[i]
                    << ", diff=" << diff
                    << ", thr=" << thr << std::endl;
        }
        ++mismatches;
      }
    }

    std::cout << "[CausalMaskGraphTest][" << scenario_name << "] mismatches=" << mismatches << std::endl;
    return mismatches == 0;
  }
};
