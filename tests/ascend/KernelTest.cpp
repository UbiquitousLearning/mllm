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

//===----------------------------------------------------------------------===//
// Element wise SUB.
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
TEST_F(AscendKernelTest, SubFloat16) {
  EXPECT_EQ(SubFloat16Test({
                {2, 3},
                {1, 1},
                {4, 4},
                {8, 8},
                {16, 16},
                {32, 32},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Element wise MUL.
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
TEST_F(AscendKernelTest, MulFloat16) {
  EXPECT_EQ(MulFloat16Test({
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
  
  // Initialize context
  mllm::initializeContext();

  // Initialize Ascend backend
  mllm::initAscendBackend();
  
  auto ret = RUN_ALL_TESTS();
  
  // Cleanup
  mllm::memoryReport();
  mllm::shutdownContext();
  
  return ret;
}

