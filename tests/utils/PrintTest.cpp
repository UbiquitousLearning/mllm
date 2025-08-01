// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "mllm/mllm.hpp"

TEST(PrintTest, Tensor) {
  mllm::initializeContext();
  using namespace mllm;  // NOLINT
  auto t = Tensor::ones({100, 24, 24}, kFloat32, kCPU);
  print(t);
  print(t.shape(), t.dtype(), t.device());
  print("---------------------------------");
  t = Tensor::ones({3, 8, 8}, kFloat32, kCPU);
  print(t);
  print(t.shape(), t.dtype(), t.device());
  mllm::memoryReport();
}
