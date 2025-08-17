// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "mllm/mllm.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

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

//===----------------------------------------------------------------------===//
// Element wise SUB.
//
// FP32
// FP16
// Int8
// Int16
// Int32
// Int64
//===----------------------------------------------------------------------===//
TEST_F(ElementwiseKernelTest, SubFloat32) {
  EXPECT_EQ(SubFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, SubFloat16) {
  EXPECT_EQ(SubFloat16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, SubInt8) {
  EXPECT_EQ(SubInt8Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, SubInt16) {
  EXPECT_EQ(SubInt16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, SubInt32) {
  EXPECT_EQ(SubInt32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Element wise MUL.
//
// FP32
// FP16
// Int8
// Int16
// Int32
// Int64
//===----------------------------------------------------------------------===//
TEST_F(ElementwiseKernelTest, MulFloat32) {
  EXPECT_EQ(MulFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, MulFloat16) {
  EXPECT_EQ(MulFloat16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, MulInt8) {
  EXPECT_EQ(MulInt8Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, MulInt16) {
  EXPECT_EQ(MulInt16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, MulInt32) {
  EXPECT_EQ(MulInt32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Element wise DIV.
//
// FP32
// FP16
// Int8
// Int16
// Int32
// Int64
//===----------------------------------------------------------------------===//
TEST_F(ElementwiseKernelTest, DivFloat32) {
  EXPECT_EQ(DivFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, DivFloat16) {
  EXPECT_EQ(DivFloat16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, DivInt8) {
  EXPECT_EQ(DivInt8Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, DivInt16) {
  EXPECT_EQ(DivInt16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, DivInt32) {
  EXPECT_EQ(DivInt32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Element wise ADD Scalar.
//
// FP32
// FP16
// Int8
// Int16
// Int32
// Int64
//===----------------------------------------------------------------------===//
TEST_F(ElementwiseKernelTest, AddScalarFloat32) {
  EXPECT_EQ(AddScalarFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, AddScalarFloat16) {
  EXPECT_EQ(AddScalarFloat16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, AddScalarInt8) {
  EXPECT_EQ(AddScalarInt8Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, AddScalarInt16) {
  EXPECT_EQ(AddScalarInt16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, AddScalarInt32) {
  EXPECT_EQ(AddScalarInt32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Element wise SUB Scalar.
//
// FP32
// FP16
// Int8
// Int16
// Int32
// Int64
//===----------------------------------------------------------------------===//
TEST_F(ElementwiseKernelTest, SubScalarFloat32) {
  EXPECT_EQ(SubScalarFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, SubScalarFloat16) {
  EXPECT_EQ(SubScalarFloat16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, SubScalarInt8) {
  EXPECT_EQ(SubScalarInt8Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, SubScalarInt16) {
  EXPECT_EQ(SubScalarInt16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, SubScalarInt32) {
  EXPECT_EQ(SubScalarInt32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Element wise MUL.
//
// FP32
// FP16
// Int8
// Int16
// Int32
// Int64
//===----------------------------------------------------------------------===//
TEST_F(ElementwiseKernelTest, MulScalarFloat32) {
  EXPECT_EQ(MulScalarFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, MulScalarFloat16) {
  EXPECT_EQ(MulScalarFloat16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, MulScalarInt8) {
  EXPECT_EQ(MulScalarInt8Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, MulScalarInt16) {
  EXPECT_EQ(MulScalarInt16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, MulScalarInt32) {
  EXPECT_EQ(MulScalarInt32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Element wise DIV.
//
// FP32
// FP16
// Int8
// Int16
// Int32
// Int64
//===----------------------------------------------------------------------===//
TEST_F(ElementwiseKernelTest, DivScalarFloat32) {
  EXPECT_EQ(DivScalarFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, DivScalarFloat16) {
  EXPECT_EQ(DivScalarFloat16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, DivScalarInt8) {
  EXPECT_EQ(DivScalarInt8Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, DivScalarInt16) {
  EXPECT_EQ(DivScalarInt16Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

TEST_F(ElementwiseKernelTest, DivScalarInt32) {
  EXPECT_EQ(DivScalarInt32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// HPC Arm SGEMV Tests
//
// D is always multiple of 32.
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "MllmBlasArmSgemvKernelTest.hpp"
TEST_F(MllmBlasArmSgemvKernelTest, matmul_fp32_gemv_nt_t_decode_small_d_qk) {
  EXPECT_EQ(test_mllm_blas_matmul_fp32_gemv_nt_t_decode_small_d_qk({
                {{"D", 64}, {"S", 1}},
                {{"D", 64}, {"S", 3}},
                {{"D", 64}, {"S", 6}},
                {{"D", 64}, {"S", 7}},
                {{"D", 128}, {"S", 7}},
            }),
            true);
}
#endif

//===----------------------------------------------------------------------===//
// LlamaFileKernelTest
//===----------------------------------------------------------------------===//
#include "LlamaFileKernelTest.hpp"
TEST_F(LlamaFileKernelTest, matmul_2) { EXPECT_EQ(oneCase({{8, 8}, {8, 8}}, false, true), true); }

TEST_F(LlamaFileKernelTest, matmul_3) { EXPECT_EQ(oneCase({{1, 8, 8}, {1, 8, 8}}, false, true), true); }

//===----------------------------------------------------------------------===//
// BlasKernelTest
//===----------------------------------------------------------------------===//
#ifdef MLLM_USE_BLAS
#include "BlasKernelTest.hpp"
TEST_F(BlasKernelTest, matmul_MxK_NxK) {
  EXPECT_EQ(matmul_MxK_NxK({
                {{"M", 64}, {"N", 64}, {"K", 64}},
            }),
            true);
}

TEST_F(BlasKernelTest, batch_matmul_BHSD) {
  EXPECT_EQ(batch_matmul_BHSD({
                {{"B", 2}, {"H", 28}, {"S", 10}, {"D", 16}},
            }),
            true);
}
#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  mllm::initializeContext();
  auto ret = RUN_ALL_TESTS();
  mllm::shutdownContext();
  mllm::memoryReport();
  return ret;
}
