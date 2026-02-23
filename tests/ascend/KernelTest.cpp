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

//===----------------------------------------------------------------------===//
// Softmax activation function.
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
#include "AscendSoftmaxKernelTest.hpp"
TEST_F(AscendSoftmaxKernelTest, SoftmaxFloat16) {
  EXPECT_EQ(SoftmaxFloat16Test({
                {2, 3},
                {1, 8},
                {4, 4},
                {8, 8},
                {16, 16},
                {1, 1024},
                {128, 128},
            },
            {-1, 0, 1}  // Test different axes
            ),
            true);
}

//===----------------------------------------------------------------------===//
// Scaled Dot-Product Attention (using existing operators).
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
#include "AscendAttentionKernelTest.hpp"
TEST_F(AscendAttentionKernelTest, ScaledDotProductAttentionFloat16) {
  EXPECT_EQ(ScaledDotProductAttentionFloat16Test({
                // {Q_shape, K_shape, V_shape}
                // Format: [B, S, D]
                {{1, 4, 8}, {1, 4, 8}, {1, 4, 8}},      // Small: B=1, S=4, D=8
                {{1, 8, 16}, {1, 8, 16}, {1, 8, 16}},   // Medium: B=1, S=8, D=16
                {{2, 4, 8}, {2, 4, 8}, {2, 4, 8}},      // Batch=2
                {{1, 16, 32}, {1, 16, 32}, {1, 16, 32}}, // Larger: B=1, S=16, D=32
                {{1, 8, 64}, {1, 8, 64}, {1, 8, 64}},   // D=64 (common head dim)
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Multi-Head Attention with Causal Mask.
//
// FP16 (Ascend currently uses FP16)
// Input format: [B, H, S, D] where H = num_heads, D = head_dim
//===----------------------------------------------------------------------===//
TEST_F(AscendAttentionKernelTest, MultiHeadAttentionFloat16) {
  EXPECT_EQ(MultiHeadAttentionFloat16Test({
                // {Q_shape, K_shape, V_shape, use_causal_mask}
                // Format: [B, H, S, D]

                // Without mask
                {{1, 1, 4, 8}, {1, 1, 4, 8}, {1, 1, 4, 8}, false},      // Single head, no mask
                {{1, 4, 8, 16}, {1, 4, 8, 16}, {1, 4, 8, 16}, false},   // 4 heads, no mask
                {{1, 8, 16, 64}, {1, 8, 16, 64}, {1, 8, 16, 64}, false}, // 8 heads, D=64

                // With causal mask
                {{1, 1, 4, 8}, {1, 1, 4, 8}, {1, 1, 4, 8}, true},       // Single head, with mask
                {{1, 4, 8, 16}, {1, 4, 8, 16}, {1, 4, 8, 16}, true},    // 4 heads, with mask
                {{1, 8, 16, 64}, {1, 8, 16, 64}, {1, 8, 16, 64}, true}, // 8 heads, with mask
                {{2, 4, 8, 32}, {2, 4, 8, 32}, {2, 4, 8, 32}, true},    // Batch=2, with mask

                // Different S_q and S_kv (useful for KV cache scenarios)
                {{1, 4, 1, 32}, {1, 4, 8, 32}, {1, 4, 8, 32}, true},    // S_q=1, S_kv=8 (decode)
                {{1, 4, 4, 32}, {1, 4, 16, 32}, {1, 4, 16, 32}, true},  // S_q < S_kv
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Grouped Query Attention (GQA).
//
// FP16 (Ascend currently uses FP16)
//===----------------------------------------------------------------------===//
TEST_F(AscendAttentionKernelTest, GroupedQueryAttentionFloat16) {
  EXPECT_EQ(GroupedQueryAttentionFloat16Test({
                // {Q_shape [B, H_q, S_q, D], K_shape [B, H_kv, S_kv, D], V_shape, use_mask}

                // GQA with 2 groups (H_q = 4, H_kv = 2)
                {{1, 4, 8, 32}, {1, 2, 8, 32}, {1, 2, 8, 32}, false},
                {{1, 4, 8, 32}, {1, 2, 8, 32}, {1, 2, 8, 32}, true},

                // GQA with 4 groups (H_q = 8, H_kv = 2)
                {{1, 8, 8, 32}, {1, 2, 8, 32}, {1, 2, 8, 32}, false},
                {{1, 8, 8, 32}, {1, 2, 8, 32}, {1, 2, 8, 32}, true},

                // MQA (Multi-Query Attention): H_kv = 1
                {{1, 4, 8, 32}, {1, 1, 8, 32}, {1, 1, 8, 32}, true},
                {{1, 8, 16, 64}, {1, 1, 16, 64}, {1, 1, 16, 64}, true},

                // Batch > 1
                {{2, 8, 8, 32}, {2, 2, 8, 32}, {2, 2, 8, 32}, true},
            }),
            true);
}

//===----------------------------------------------------------------------===//
// Concat
//===----------------------------------------------------------------------===//
#include "AscendConcatKernelTest.hpp"
TEST_F(AscendConcatKernelTest, ConcatFloat16) {
  EXPECT_EQ(ConcatFloat16Test({{2, 3}, {2, 3}}, 0), true);
  EXPECT_EQ(ConcatFloat16Test({{1, 8}, {1, 8}}, 1), true);
  EXPECT_EQ(ConcatFloat16Test({{4, 16}, {4, 16}, {4, 16}}, 0), true);
  EXPECT_EQ(ConcatFloat16Test({{2, 3, 4}, {2, 3, 5}}, 2), true);
  EXPECT_EQ(ConcatFloat16Test({{2, 3, 4}, {2, 3, 6}}, -1), true);
  EXPECT_EQ(ConcatFloat16Test({{2, 7}}, 0), true);
}

//===----------------------------------------------------------------------===//
// Slice
//===----------------------------------------------------------------------===//
#include "AscendSliceKernelTest.hpp"
TEST_F(AscendSliceKernelTest, SliceFloat16) {
  using namespace mllm;
  // SliceIndicesPair(start, end)
  EXPECT_EQ(SliceFloat16Test({4, 4}, {SliceIndicesPair(0, 2), SliceIndicesPair(0, 4)}), true);
  EXPECT_EQ(SliceFloat16Test({4, 8}, {SliceIndicesPair(1, 3), SliceIndicesPair(2, 6)}), true);
  EXPECT_EQ(SliceFloat16Test({2, 16}, {SliceIndicesPair(0, 1), SliceIndicesPair(0, 8)}), true);
  EXPECT_EQ(SliceFloat16Test({5, 4}, {SliceIndicesPair(-3, -1), SliceIndicesPair(0, 4)}), true);
  EXPECT_EQ(SliceFloat16Test({3, 4, 5}, {SliceIndicesPair(kAll, kAll), SliceIndicesPair(1, 3), SliceIndicesPair(0, 5)}), true);
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

