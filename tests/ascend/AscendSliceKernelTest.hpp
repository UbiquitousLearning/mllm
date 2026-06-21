// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Functional.hpp"
#include "KernelTestHelper.hpp"

class AscendSliceKernelTest : public KernelTest {
 public:
  bool SliceFloat16Test(mllm::Tensor::shape_t input_shape, mllm::SliceIndices indices) {
    using namespace mllm;

    Tensor in_cpu = Tensor::random(input_shape, -1.0, 1.0, kFloat16, kCPU);

    // CPU Reference (View)
    Tensor out_cpu = in_cpu[indices];

    // Ascend (Copy/Kernel)
    Tensor in_ascend = in_cpu.to(kAscend);
    Tensor out_ascend = in_ascend[indices];
    Tensor out_back = out_ascend.to(kCPU);

    // The output from Ascend should match the view on CPU
    // We compare them. Note: out_cpu might be non-contiguous, allClose should handle or we make it contiguous.
    Tensor out_cpu_cont = out_cpu.contiguous();
    
    auto result = test::allClose(out_back, out_cpu_cont, 1e-2, 1e-2);
    if (!result.is_close) {
        std::cout << "[SliceTest] FAILED!" << std::endl;
        return false;
    }
    std::cout << "[SliceTest] PASSED" << std::endl;
    return true;
  }
};
