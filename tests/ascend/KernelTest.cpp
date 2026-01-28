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
// SiLU activation function.
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
#include "AscendSiLUKernelTest.hpp"
TEST_F(AscendSiLUKernelTest, SiLUFloat16) {
  EXPECT_EQ(SiLUFloat16Test({
                {2, 3},
                {1, 1},
                {4, 4},
                {8, 8},
                {16, 16},
                {32, 32},
                {1, 1024},
                {128, 128},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Linear layer (MatMul based test).
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
#include "AscendLinearKernelTest.hpp"
TEST_F(AscendLinearKernelTest, LinearFloat16) {
  EXPECT_EQ(LinearFloat16Test({
                // {input_shape, in_channels, out_channels}
                {{2, 3}, 3, 4},
                {{1, 8}, 8, 16},
                {{4, 16}, 16, 32},
                {{8, 32}, 32, 64},
                {{1, 1024}, 1024, 512},
            }),
            true);
}

TEST_F(AscendLinearKernelTest, LinearWithBiasFloat16) {
  EXPECT_EQ(LinearWithBiasFloat16Test({
                // {input_shape, in_channels, out_channels}
                {{2, 3}, 3, 4},
                {{1, 8}, 8, 16},
                {{4, 16}, 16, 32},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// RMSNorm layer.
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
#include "AscendRMSNormKernelTest.hpp"
TEST_F(AscendRMSNormKernelTest, RMSNormFloat16) {
  EXPECT_EQ(RMSNormFloat16Test({
                // {input_shape, norm_size, epsilon}
                // Note: ATB RMSNorm requires last dim to be multiple of 16 (FP16 alignment)
                {{2, 16}, 16, 1e-5f},
                {{1, 32}, 32, 1e-5f},
                {{4, 64}, 64, 1e-6f},
                {{8, 128}, 128, 1e-5f},
                {{1, 1024}, 1024, 1e-5f},
                {{128, 256}, 256, 1e-5f},
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

