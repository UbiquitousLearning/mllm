// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <unordered_map>

#include "mllm/mllm.hpp"
#include "mllm/backends/cpu/kernels/arm/gelu.hpp"

#include "KernelTestHelper.hpp"

class GELUKernelTest : public KernelTest {
 public:
  GELUKernelTest() = default;
  ~GELUKernelTest() override = default;

  bool cmp(std::unordered_map<std::string, int32_t>& vars) {
    int S = vars["S"];

    auto A = mllm::Tensor::random({S}, mllm::kFloat32, mllm::kCPU);
    auto refDst = mllm::Tensor::ones({S}, mllm::kFloat32, mllm::kCPU);
    auto Dst = mllm::Tensor::ones({S}, mllm::kFloat32, mllm::kCPU);

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
    mllm::cpu::arm::gelu_fp32(refDst.ptr<float>(), A.ptr<float>(), S, 1);
    mllm::cpu::arm::gelu_fp32(Dst.ptr<float>(), A.ptr<float>(), S, 4);

    auto result = mllm::test::allClose(refDst, Dst);
    if (!result.is_close) {
      mllm::print(result);
      return false;
    }
#endif
    return true;
  }

  bool test_cmp(const std::vector<std::unordered_map<std::string, int32_t>>& vars) {
    for (auto v : vars) {
      if (!cmp(v)) { return false; }
    }
    return true;
  }
};
