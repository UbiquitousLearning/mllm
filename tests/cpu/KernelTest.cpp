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
// CausalMaskOp
//===----------------------------------------------------------------------===//
#include "CausalMaskOpTest.hpp"
TEST_F(CausalMaskOpTest, PrefillScenario) {
  auto result = runScenario(1, 1, 4, 4);
  EXPECT_TRUE(result.is_close);
}

TEST_F(CausalMaskOpTest, DecodeScenario) {
  auto result = runScenario(1, 1, 1, 6);
  EXPECT_TRUE(result.is_close);
}

TEST_F(CausalMaskOpTest, AppendScenario) {
  auto result = runScenario(2, 3, 3, 7);
  EXPECT_TRUE(result.is_close);
}

//===----------------------------------------------------------------------===//
// GELU
//===----------------------------------------------------------------------===//
#include "tests/cpu/GELUKernelTest.hpp"
TEST_F(GELUKernelTest, test_precision_bt_1threads_4threads) {
  EXPECT_EQ(test_cmp({
                {{"S", 10}},
                {{"S", 128}},
                {{"S", 256}},
                {{"S", 18}},
                {{"S", 128600}},
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

TEST_F(MllmBlasArmSgemvKernelTest, matmul_fp32_gemv_nt_nt_decode_small_d_wv) {
  EXPECT_EQ(test_mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv({
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
// HPC Arm SGEMV Tests
//
// D is always multiple of 32.
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "MllmBlasArmSgemmKernelTest.hpp"
TEST_F(MllmBlasArmSgemmKernelTest, test_mllm_blas_matmul_fp32_gemm_nt_nt) {
  EXPECT_EQ(test_mllm_blas_matmul_fp32_gemm_nt_nt({
                {{"D", 64}, {"S_Q", 1}, {"S_KV", 4}},  // Fallback to  test_mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv
                {{"D", 64}, {"S_Q", 32}, {"S_KV", 16}},
                {{"D", 64}, {"S_Q", 33}, {"S_KV", 17}},
                {{"D", 64}, {"S_Q", 3}, {"S_KV", 4}},
                {{"D", 64}, {"S_Q", 4}, {"S_KV", 16}},
                {{"D", 64}, {"S_Q", 8}, {"S_KV", 16}},
                {{"D", 125}, {"S_Q", 5}, {"S_KV", 15}},
                {{"D", 128}, {"S_Q", 1}, {"S_KV", 20}},  // Fallback to  test_mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv
                {{"D", 100}, {"S_Q", 464}, {"S_KV", 513}},
            }),
            true);
}

TEST_F(MllmBlasArmSgemmKernelTest, test_mllm_blas_matmul_fp32_gemm_nt_t) {
  EXPECT_EQ(test_mllm_blas_matmul_fp32_gemm_nt_t({
                {{"batch", 2},
                 {"in_channels", 128},
                 {"out_channels", 128}},  // Fallback to  test_mllm_blas_matmul_fp32_gemv_nt_nt_decode_small_d_wv
                {{"batch", 1}, {"in_channels", 64}, {"out_channels", 128}},
                {{"batch", 1}, {"in_channels", 5}, {"out_channels", 125}},
                {{"batch", 2}, {"in_channels", 5}, {"out_channels", 125}},
                {{"batch", 2}, {"in_channels", 5}, {"out_channels", 15}},
                {{"batch", 680}, {"in_channels", 1280}, {"out_channels", 3420}},
                {{"batch", 680}, {"in_channels", 3420}, {"out_channels", 1280}},
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

//===----------------------------------------------------------------------===//
// TransposeTest
//===----------------------------------------------------------------------===//
#include "TransposeKernelTest.hpp"
TEST_F(TransposeKernelTest, HWTransposition) { EXPECT_EQ(testHWTransposition(), true); }

TEST_F(TransposeKernelTest, BSHDTransposition) { EXPECT_EQ(testBSHDTransposition(), true); }

TEST_F(TransposeKernelTest, GeneralTransposition) { EXPECT_EQ(testGeneralTransposition(), true); }

//===----------------------------------------------------------------------===//
// Permute operation tests
//===----------------------------------------------------------------------===//
#include "PermuteKernelTest.hpp"
TEST_F(PermuteKernelTest, Permute2DAndHigher) { EXPECT_EQ(test2DPermutation(), true); }

//===----------------------------------------------------------------------===//
// TopK operation tests
//===----------------------------------------------------------------------===//
#include "TopKKernelTest.hpp"
TEST_F(TopKKernelTest, TopKTest) { EXPECT_EQ(testTopK({{10}, {1, 10}, {5, 10}, {2, 5, 10}, {1, 4, 8, 16}}, 3), true); }

TEST_F(TopKKernelTest, TopKTestDim) { EXPECT_EQ(testTopK({{2, 5, 10}}, 4, 1), true); }

//===----------------------------------------------------------------------===//
// Clip operation tests
//===----------------------------------------------------------------------===//
#include "ClipKernelTest.hpp"
TEST_F(ClipKernelTest, ClipTest) {
  EXPECT_EQ(testClip({{10}, {1, 10}, {5, 10}, {2, 5, 10}, {1, 4, 8, 16}}, -5.0f, 5.0f), true);
}

//===----------------------------------------------------------------------===//
// Reduce MEAN.
//
// FP32
// FP16
//===----------------------------------------------------------------------===//
#include "ReduceKernelTest.hpp"
TEST_F(ReduceKernelTest, MeanFloat32) {
  EXPECT_EQ(ReduceMeanFloat32Test({
                {3},
                {9},
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
TEST_F(ReduceKernelTest, MeanFloat16) {
  EXPECT_EQ(ReduceMeanFloat16Test({
                {3},
                {9},
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}
#endif

//===----------------------------------------------------------------------===//
// Reduce SUM.
//
// FP32
// FP16
//===----------------------------------------------------------------------===//
TEST_F(ReduceKernelTest, SumFloat32) {
  EXPECT_EQ(ReduceSumFloat32Test({
                {3},
                {9},
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
// FIXME:
// This kernel has precision issues !!!
// This kernel has precision issues !!!
// This kernel has precision issues !!!
// TEST_F(ReduceKernelTest, SumFloat16) {
//   EXPECT_EQ(ReduceSumFloat16Test({
//                 {3},
//                 {9},
//                 {42},
//                 {5, 5},
//                 {16, 16},
//                 {16, 18},
//                 {32, 32},
//                 {128, 128, 128},
//             }),
//             true);
// }
#endif

//===----------------------------------------------------------------------===//
// Scatter 2 Shards Attn
//===----------------------------------------------------------------------===//
#include "Scatter2ShardsKernelTest.hpp"
TEST_F(Scatter2ShardsKernelTest, one) { EXPECT_EQ(testScatter2Shards(), true); }

//===----------------------------------------------------------------------===//
// Paged Attn
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "PagedAttnTest.hpp"
TEST_F(PagedAttnTest, fwd_bshd) {
  EXPECT_EQ(manyCases({
                // Query Shape, KV Shape
                {mllm::Tensor::shape_t{1, 10, 28, 128}, mllm::Tensor::shape_t{1, 10, 2, 128}},
                {mllm::Tensor::shape_t{1, 10, 28, 64}, mllm::Tensor::shape_t{1, 10, 28, 64}},
                {mllm::Tensor::shape_t{1, 1, 28, 64}, mllm::Tensor::shape_t{1, 10, 2, 64}},
                {mllm::Tensor::shape_t{1, 3, 28, 64}, mllm::Tensor::shape_t{1, 10, 2, 64}},
            }),
            true);
}
#endif

//===----------------------------------------------------------------------===//
// Radix Attn
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "RadixAttnKernel.hpp"
TEST_F(RadixAttnKernelTest, fwd_bshd) {
  EXPECT_EQ(testRadixAttn({{
                               {"H_Q", 28},
                               {"H_KV", 2},
                               {"S_Q", 10},
                               {"S_KV", 10},
                               {"D", 128},
                           },
                           {
                               {"H_Q", 28},
                               {"H_KV", 2},
                               {"S_Q", 10},
                               {"S_KV", 20},
                               {"D", 128},
                           },
                           {
                               {"H_Q", 28},
                               {"H_KV", 2},
                               {"S_Q", 1},
                               {"S_KV", 20},
                               {"D", 128},
                           }}),
            true);
}
#endif

//===----------------------------------------------------------------------===//
// Radix Attn
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "FlashAttentionKernelTest.hpp"
TEST_F(FlashAttn2KernelTest, fwd_bshd) {
  EXPECT_EQ(testRadixAttn({{
                               {"H_Q", 28},
                               {"H_KV", 2},
                               {"S_Q", 10},
                               {"S_KV", 10},
                               {"D", 128},
                           },
                           {
                               {"H_Q", 28},
                               {"H_KV", 2},
                               {"S_Q", 10},
                               {"S_KV", 20},
                               {"D", 128},
                           },
                           {
                               {"H_Q", 28},
                               {"H_KV", 2},
                               {"S_Q", 1},
                               {"S_KV", 20},
                               {"D", 128},
                           }}),
            true);
}
#endif

//===----------------------------------------------------------------------===//
// Conv2D Test
//
// auto in_channel = cfg.at("in_channel");
// auto out_channel = cfg.at("out_channel");
// auto I_H = cfg.at("I_H");
// auto I_W = cfg.at("I_W");
// auto K_H = cfg.at("K_H");
// auto K_W = cfg.at("K_W");
// auto S_H = cfg.at("S_H");
// auto S_W = cfg.at("S_W");
// auto P_H = cfg.at("P_H");
// auto P_W = cfg.at("P_W");
// auto bias = cfg.at("bias");
//
// In deepseek-ocr we have
// CASE 1:
//  in_channel = 3
//  out_channel = 1024
//  I_H = 224
//  I_W = 224
//  K_H = 14
//  K_W = 14
//  S_H = 14
//  S_W = 14
//  P_H = 0
//  P_W = 0
//  bias = false
// CASE 2:
//
//===----------------------------------------------------------------------===//
#include "Conv2DKernelTest.hpp"
TEST_F(Conv2DKernelTest, im2col) {
  EXPECT_EQ(testConv2D({
                // CLIP patch embedding
                {
                    {"in_channel", 3},
                    {"out_channel", 1024},
                    {"I_H", 224},
                    {"I_W", 224},
                    {"K_H", 14},
                    {"K_W", 14},
                    {"S_H", 14},
                    {"S_W", 14},
                    {"P_H", 0},
                    {"P_W", 0},
                    {"bias", 0},
                },
                // SAM PatchEmbed.proj
                {
                    {"in_channel", 3},
                    {"out_channel", 768},
                    {"I_H", 1024},
                    {"I_W", 1024},
                    {"K_H", 16},
                    {"K_W", 16},
                    {"S_H", 16},
                    {"S_W", 16},
                    {"P_H", 0},
                    {"P_W", 0},
                    {"bias", 1},
                },
                // neck: Conv2D(768 -> 12, 1x1, stride=1, pad=0, bias=false)
                {
                    {"in_channel", 768},
                    {"out_channel", 12},
                    {"I_H", 64},
                    {"I_W", 64},
                    {"K_H", 1},
                    {"K_W", 1},
                    {"S_H", 1},
                    {"S_W", 1},
                    {"P_H", 0},
                    {"P_W", 0},
                    {"bias", 0},
                },
                // neck: Conv2D(256 -> 256, 3x3, stride=1, pad=1, bias=false)
                {
                    {"in_channel", 256},
                    {"out_channel", 256},
                    {"I_H", 64},
                    {"I_W", 64},
                    {"K_H", 3},
                    {"K_W", 3},
                    {"S_H", 1},
                    {"S_W", 1},
                    {"P_H", 1},
                    {"P_W", 1},
                    {"bias", 0},
                },
                // net_2_: Conv2D(256 -> 512, 3x3, stride=2, pad=1, bias=false)
                {
                    {"in_channel", 256},
                    {"out_channel", 512},
                    {"I_H", 64},
                    {"I_W", 64},
                    {"K_H", 3},
                    {"K_W", 3},
                    {"S_H", 2},
                    {"S_W", 2},
                    {"P_H", 1},
                    {"P_W", 1},
                    {"bias", 0},
                },
                // net_3_: Conv2D(512 -> 1024, 3x3, stride=2, pad=1, bias=false)
                {
                    {"in_channel", 512},
                    {"out_channel", 1024},
                    {"I_H", 32},
                    {"I_W", 32},
                    {"K_H", 3},
                    {"K_W", 3},
                    {"S_H", 2},
                    {"S_W", 2},
                    {"P_H", 1},
                    {"P_W", 1},
                    {"bias", 0},
                },
            }),
            true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  mllm::initializeContext();
  { auto ret = RUN_ALL_TESTS(); }
  mllm::memoryReport();
  mllm::shutdownContext();
  return 0;
}
