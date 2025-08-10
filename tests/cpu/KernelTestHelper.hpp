// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <gtest/gtest.h>

class KernelTest : public testing::Test {
 public:
  KernelTest() = default;
  ~KernelTest() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  void SetUp() override {}

  void TearDown() override {}
};