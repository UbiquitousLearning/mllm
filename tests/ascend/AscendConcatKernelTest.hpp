// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Functional.hpp"
#include "KernelTestHelper.hpp" // Has KernelTest base class

class AscendConcatKernelTest : public KernelTest {
 public:
  bool ConcatFloat16Test(const std::vector<mllm::Tensor::shape_t>& input_shapes, int dim) {
    using namespace mllm;

    std::vector<Tensor> inputs_cpu;
    for (const auto& shape : input_shapes) {
      inputs_cpu.push_back(Tensor::random(shape, -1.0, 1.0, kFloat16, kCPU));
    }

    // CPU Reference
    auto out_cpu = nn::functional::concat(inputs_cpu, dim);

    // Ascend
    std::vector<Tensor> inputs_ascend;
    for (auto& t : inputs_cpu) {
      inputs_ascend.push_back(t.to(kAscend));
    }

    auto out_ascend = nn::functional::concat(inputs_ascend, dim);
    auto out_back = out_ascend.to(kCPU);

    auto result = test::allClose(out_back, out_cpu, 1e-2, 1e-2);
    if (!result.is_close) {
        std::cout << "[ConcatTest] FAILED! dim=" << dim << std::endl;
        return false;
    }
    std::cout << "[ConcatTest] PASSED dim=" << dim << std::endl;
    return true;
  }
};
