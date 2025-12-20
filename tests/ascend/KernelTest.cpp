// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "mllm/mllm.hpp"

/// Kernel tests
#include "AscendKernelTest.hpp"

//===----------------------------------------------------------------------===//
// Element wise ADD.
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
TEST_F(AscendKernelTest, AddFloat16) {
  EXPECT_EQ(AddFloat16Test({
                {2, 3},
                {1, 1},
                {4, 4},
                {8, 8},
                {16, 16},
                {32, 32},
            }),
            true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  
  // Initialize Ascend backend
  mllm::initAscendBackend();
  
  // Initialize context
  mllm::initializeContext();
  
  auto ret = RUN_ALL_TESTS();
  
  // Cleanup
  mllm::memoryReport();
  mllm::shutdownContext();
  
  return ret;
}

