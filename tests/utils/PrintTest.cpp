/**
 * @file PrintTest.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <gtest/gtest.h>

#include "mllm/mllm.hpp"

TEST(PrintTest, Tensor) {
  using namespace mllm;  // NOLINT
  auto t = Tensor::ones({8, 8}, kFloat16, kCPU);
  print(t);
  print(t.shape(), t.dtype(), t.device());
}
