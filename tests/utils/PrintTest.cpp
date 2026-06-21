// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "mllm/mllm.hpp"
#include "mllm/utils/ProgressBar.hpp"

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
  {
    mllm::ProgressBar progress_bar("Testing Progress", 100);
    for (int i = 0; i < 100; i++) { progress_bar.update(i); }
  }
  mllm::memoryReport();
}
