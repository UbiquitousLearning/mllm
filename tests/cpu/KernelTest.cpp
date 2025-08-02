// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "mllm/mllm.hpp"

/// Kernel tests
#include "ElementwiseKernelTest.hpp"

//===----------------------------------------------------------------------===//
// Element wise ADD.
//
// FP32
// FP16
// Int8
// Int16
// Int32
// Int64
//===----------------------------------------------------------------------===//
TEST_F(ElementwiseKernelTest, AddFloat32) {
  EXPECT_EQ(AddFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, AddFloat16) {
  EXPECT_EQ(AddFloat16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, AddInt8) {
  EXPECT_EQ(AddInt8Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, AddInt16) {
  EXPECT_EQ(AddInt16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, AddInt32) {
  EXPECT_EQ(AddInt32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  mllm::initializeContext();
  auto ret = RUN_ALL_TESTS();
  mllm::shutdownContext();
  mllm::memoryReport();
  return ret;
}