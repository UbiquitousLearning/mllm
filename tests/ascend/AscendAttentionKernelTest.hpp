// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Functional.hpp"
#include "KernelTestHelper.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"
#include <vector>
#include <cmath>
#include <limits>

class AscendAttentionKernelTest : public KernelTest {
 public:
  AscendAttentionKernelTest() = default;
  ~AscendAttentionKernelTest() override = default;

  // Test Scaled Dot-Product Attention using existing operators
  // Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
  bool ScaledDotProductAttentionFloat16Test(const std::vector<std::tuple<mllm::Tensor::shape_t, mllm::Tensor::shape_t, mllm::Tensor::shape_t>>& test_cases) {
    using namespace mllm;  // NOLINT

    for (const auto& [q_shape, k_shape, v_shape] : test_cases) {
      // Validate shapes: Q=[B, S_q, D], K=[B, S_kv, D], V=[B, S_kv, D]
      MLLM_RT_ASSERT_EQ(q_shape.size(), 3);
      MLLM_RT_ASSERT_EQ(k_shape.size(), 3);
      MLLM_RT_ASSERT_EQ(v_shape.size(), 3);
      MLLM_RT_ASSERT_EQ(q_shape[0], k_shape[0]);  // Same batch size
      MLLM_RT_ASSERT_EQ(q_shape[0], v_shape[0]);
      MLLM_RT_ASSERT_EQ(q_shape[2], k_shape[2]);  // Same D dimension
      MLLM_RT_ASSERT_EQ(k_shape[1], v_shape[1]);  // K and V have same sequence length

      int32_t B = static_cast<int32_t>(q_shape[0]);
      int32_t S_q = static_cast<int32_t>(q_shape[1]);
      int32_t S_kv = static_cast<int32_t>(k_shape[1]);
      int32_t D = static_cast<int32_t>(q_shape[2]);

      // 1. Create random FP16 inputs on CPU
      Tensor Q_cpu = Tensor::random(q_shape, -1.0f, 1.0f, kFloat16, kCPU);
      Tensor K_cpu = Tensor::random(k_shape, -1.0f, 1.0f, kFloat16, kCPU);
      Tensor V_cpu = Tensor::random(v_shape, -1.0f, 1.0f, kFloat16, kCPU);

      // 2. Compute reference result on CPU using FP32 for better precision
      Tensor Q_cpu_fp32 = Tensor::zeros(q_shape, kFloat32, kCPU);
      Tensor K_cpu_fp32 = Tensor::zeros(k_shape, kFloat32, kCPU);
      Tensor V_cpu_fp32 = Tensor::zeros(v_shape, kFloat32, kCPU);

      // Convert FP16 to FP32
      {
        auto* q_fp16 = Q_cpu.ptr<mllm_fp16_t>();
        auto* k_fp16 = K_cpu.ptr<mllm_fp16_t>();
        auto* v_fp16 = V_cpu.ptr<mllm_fp16_t>();
        auto* q_fp32 = Q_cpu_fp32.ptr<mllm_fp32_t>();
        auto* k_fp32 = K_cpu_fp32.ptr<mllm_fp32_t>();
        auto* v_fp32 = V_cpu_fp32.ptr<mllm_fp32_t>();

        for (size_t i = 0; i < Q_cpu.numel(); ++i) {
          q_fp32[i] = MLLM_FP16_TO_FP32(q_fp16[i]);
        }
        for (size_t i = 0; i < K_cpu.numel(); ++i) {
          k_fp32[i] = MLLM_FP16_TO_FP32(k_fp16[i]);
        }
        for (size_t i = 0; i < V_cpu.numel(); ++i) {
          v_fp32[i] = MLLM_FP16_TO_FP32(v_fp16[i]);
        }
      }

      // Compute reference attention on CPU (FP32)
      Tensor ref_cpu_fp32 = Tensor::zeros({B, S_q, D}, kFloat32, kCPU);
      {
        auto* q_ptr = Q_cpu_fp32.ptr<mllm_fp32_t>();
        auto* k_ptr = K_cpu_fp32.ptr<mllm_fp32_t>();
        auto* v_ptr = V_cpu_fp32.ptr<mllm_fp32_t>();
        auto* out_ptr = ref_cpu_fp32.ptr<mllm_fp32_t>();

        float scale = 1.0f / std::sqrt(static_cast<float>(D));

        for (int32_t b = 0; b < B; ++b) {
          // Compute Q @ K^T for this batch
          std::vector<float> scores(S_q * S_kv, 0.0f);

          for (int32_t i = 0; i < S_q; ++i) {
            for (int32_t j = 0; j < S_kv; ++j) {
              float sum = 0.0f;
              for (int32_t k = 0; k < D; ++k) {
                float q_val = q_ptr[b * S_q * D + i * D + k];
                float k_val = k_ptr[b * S_kv * D + j * D + k];
                sum += q_val * k_val;
              }
              scores[i * S_kv + j] = sum * scale;
            }
          }

          // Apply softmax along the last dimension (S_kv)
          std::vector<float> attn_weights(S_q * S_kv);
          for (int32_t i = 0; i < S_q; ++i) {
            // Find max for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int32_t j = 0; j < S_kv; ++j) {
              max_val = std::max(max_val, scores[i * S_kv + j]);
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int32_t j = 0; j < S_kv; ++j) {
              float exp_val = std::exp(scores[i * S_kv + j] - max_val);
              attn_weights[i * S_kv + j] = exp_val;
              sum_exp += exp_val;
            }

            // Normalize
            for (int32_t j = 0; j < S_kv; ++j) {
              attn_weights[i * S_kv + j] /= sum_exp;
            }
          }

          // Compute output: attn_weights @ V
          // out[S_q, D] = attn_weights[S_q, S_kv] @ V[S_kv, D]
          for (int32_t i = 0; i < S_q; ++i) {
            for (int32_t k = 0; k < D; ++k) {
              float sum = 0.0f;
              for (int32_t j = 0; j < S_kv; ++j) {
                float attn_val = attn_weights[i * S_kv + j];
                float v_val = v_ptr[b * S_kv * D + j * D + k];
                sum += attn_val * v_val;
              }
              out_ptr[b * S_q * D + i * D + k] = sum;
            }
          }
        }
      }

      // Convert reference back to FP16
      Tensor ref_cpu = Tensor::zeros({B, S_q, D}, kFloat16, kCPU);
      {
        auto* ref_fp32 = ref_cpu_fp32.ptr<mllm_fp32_t>();
        auto* ref_fp16 = ref_cpu.ptr<mllm_fp16_t>();
        for (size_t i = 0; i < ref_cpu.numel(); ++i) {
          ref_fp16[i] = MLLM_FP32_TO_FP16(ref_fp32[i]);
        }
      }

      // 3. Move inputs to Ascend and compute attention using existing operators
      auto Q_ascend = Q_cpu.to(kAscend);
      auto K_ascend = K_cpu.to(kAscend);
      auto V_ascend = V_cpu.to(kAscend);

      float scale = 1.0f / std::sqrt(static_cast<float>(D));

      // Step 1: Q @ K^T (transpose_b=true)
      auto scores = mllm::nn::functional::matmul(Q_ascend, K_ascend, false, true);

      // Step 2: Scale by 1/sqrt(d_k)
      auto scale_tensor_cpu = Tensor::ones({1}, kFloat16, kCPU) * scale;
      auto scale_tensor = scale_tensor_cpu.to(kAscend);
      auto scaled_scores = scores * scale_tensor;

      // Step 3: Softmax along last dimension
      auto attn_weights = mllm::nn::functional::softmax(scaled_scores, -1);

      // Step 4: attn_weights @ V
      auto output_ascend = mllm::nn::functional::matmul(attn_weights, V_ascend, false, false);

      // 4. Move result back to CPU and compare
      auto output_cpu = output_ascend.to(kCPU);

      auto result = mllm::test::allClose(output_cpu, ref_cpu, 5e-2f, 5e-2f);
      if (!result.is_close) {
        MLLM_ERROR("Attention test failed for shape Q=[{},{},{}], K=[{},{},{}], V=[{},{},{}]",
                   q_shape[0], q_shape[1], q_shape[2],
                   k_shape[0], k_shape[1], k_shape[2],
                   v_shape[0], v_shape[1], v_shape[2]);
        MLLM_ERROR("Max absolute diff: {}, Max relative diff: {}",
                   result.max_absolute_diff, result.max_relative_diff);
        return false;
      }
    }
    return true;
  }

  //===----------------------------------------------------------------------===//
  // Multi-Head Attention with optional Causal Mask
  //
  // Input shapes: Q=[B, H, S_q, D], K=[B, H, S_kv, D], V=[B, H, S_kv, D]
  // where H = num_heads, D = head_dim
  // Mask shape: [1, 1, S_q, S_kv] (broadcastable to [B, H, S_q, S_kv])
  //
  // Attention(Q, K, V, mask) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
  //===----------------------------------------------------------------------===//
  bool MultiHeadAttentionFloat16Test(
      const std::vector<std::tuple<
          mllm::Tensor::shape_t,  // Q shape [B, H, S_q, D]
          mllm::Tensor::shape_t,  // K shape [B, H, S_kv, D]
          mllm::Tensor::shape_t,  // V shape [B, H, S_kv, D]
          bool                     // use_causal_mask
      >>& test_cases) {
    using namespace mllm;  // NOLINT

    for (const auto& [q_shape, k_shape, v_shape, use_mask] : test_cases) {
      // Validate shapes: Q=[B, H, S_q, D], K=[B, H, S_kv, D], V=[B, H, S_kv, D]
      MLLM_RT_ASSERT_EQ(q_shape.size(), 4);
      MLLM_RT_ASSERT_EQ(k_shape.size(), 4);
      MLLM_RT_ASSERT_EQ(v_shape.size(), 4);
      MLLM_RT_ASSERT_EQ(q_shape[0], k_shape[0]);  // Same batch size
      MLLM_RT_ASSERT_EQ(q_shape[0], v_shape[0]);
      MLLM_RT_ASSERT_EQ(q_shape[1], k_shape[1]);  // Same num_heads
      MLLM_RT_ASSERT_EQ(q_shape[1], v_shape[1]);
      MLLM_RT_ASSERT_EQ(q_shape[3], k_shape[3]);  // Same head_dim
      MLLM_RT_ASSERT_EQ(k_shape[2], v_shape[2]);  // K and V have same sequence length

      int32_t B = static_cast<int32_t>(q_shape[0]);
      int32_t H = static_cast<int32_t>(q_shape[1]);  // num_heads
      int32_t S_q = static_cast<int32_t>(q_shape[2]);
      int32_t S_kv = static_cast<int32_t>(k_shape[2]);
      int32_t D = static_cast<int32_t>(q_shape[3]);  // head_dim

      // 1. Create random FP16 inputs on CPU
      Tensor Q_cpu = Tensor::random(q_shape, -0.5f, 0.5f, kFloat16, kCPU);
      Tensor K_cpu = Tensor::random(k_shape, -0.5f, 0.5f, kFloat16, kCPU);
      Tensor V_cpu = Tensor::random(v_shape, -0.5f, 0.5f, kFloat16, kCPU);

      // 2. Create causal mask if needed
      // Causal mask: mask[i, j] = 0 if j <= i, else -inf (large negative value)
      Tensor mask_cpu;
      if (use_mask) {
        mask_cpu = Tensor::zeros({1, 1, S_q, S_kv}, kFloat16, kCPU);
        auto* mask_ptr = mask_cpu.ptr<mllm_fp16_t>();

        // Fill causal mask: upper triangular part is masked (-inf)
        for (int32_t i = 0; i < S_q; ++i) {
          for (int32_t j = 0; j < S_kv; ++j) {
            int32_t offset = S_kv - S_q;
            if (j > i + offset) {
              mask_ptr[i * S_kv + j] = MLLM_FP32_TO_FP16(-10000.0f);
            }
          }
        }
      }

      // 3. Compute reference result on CPU using FP32 for better precision
      Tensor ref_cpu = computeMultiHeadAttentionCPU(Q_cpu, K_cpu, V_cpu, mask_cpu, use_mask);

      // 4. Move inputs to Ascend and compute attention
      auto Q_ascend = Q_cpu.to(kAscend);
      auto K_ascend = K_cpu.to(kAscend);
      auto V_ascend = V_cpu.to(kAscend);

      float scale = 1.0f / std::sqrt(static_cast<float>(D));

      // Step 1: Q @ K^T (transpose_b=true)
      auto scores = mllm::nn::functional::matmul(Q_ascend, K_ascend, false, true);

      // Step 2: Scale by 1/sqrt(d_k)
      auto scale_tensor_cpu = Tensor::ones({1}, kFloat16, kCPU);
      {
        auto* scale_ptr = scale_tensor_cpu.ptr<mllm_fp16_t>();
        scale_ptr[0] = MLLM_FP32_TO_FP16(scale);
      }
      auto scale_tensor = scale_tensor_cpu.to(kAscend);
      auto scaled_scores = scores * scale_tensor;

      // Step 3: Add mask if needed (broadcasting: [1, 1, S_q, S_kv] -> [B, H, S_q, S_kv])
      if (use_mask) {
        auto mask_ascend = mask_cpu.to(kAscend);
        scaled_scores = scaled_scores + mask_ascend;
      }

      // Step 4: Softmax along last dimension
      auto attn_weights = mllm::nn::functional::softmax(scaled_scores, -1);

      // Step 5: attn_weights @ V
      // [B, H, S_q, S_kv] @ [B, H, S_kv, D] -> [B, H, S_q, D]
      auto output_ascend = mllm::nn::functional::matmul(attn_weights, V_ascend, false, false);

      // 5. Move result back to CPU and compare
      auto output_cpu = output_ascend.to(kCPU);

      auto result = mllm::test::allClose(output_cpu, ref_cpu, 5e-2f, 5e-2f);
      if (!result.is_close) {
        MLLM_ERROR("Multi-head attention test failed for shape Q=[{},{},{},{}], K=[{},{},{},{}], V=[{},{},{},{}], mask={}",
                   q_shape[0], q_shape[1], q_shape[2], q_shape[3],
                   k_shape[0], k_shape[1], k_shape[2], k_shape[3],
                   v_shape[0], v_shape[1], v_shape[2], v_shape[3],
                   use_mask ? "true" : "false");
        MLLM_ERROR("Max absolute diff: {}, Max relative diff: {}",
                   result.max_absolute_diff, result.max_relative_diff);
        return false;
      }

      MLLM_INFO("Multi-head attention test passed: B={}, H={}, S_q={}, S_kv={}, D={}, mask={}",
                B, H, S_q, S_kv, D, use_mask ? "true" : "false");
    }
    return true;
  }

  //===----------------------------------------------------------------------===//
  // Multi-Head Attention with Grouped Query Attention (GQA) support
  //
  // GQA: num_q_heads > num_kv_heads, each KV head is shared by multiple Q heads
  // Input shapes: Q=[B, H_q, S_q, D], K=[B, H_kv, S_kv, D], V=[B, H_kv, S_kv, D]
  //===----------------------------------------------------------------------===//
  bool GroupedQueryAttentionFloat16Test(
      const std::vector<std::tuple<
          mllm::Tensor::shape_t,  // Q shape [B, H_q, S_q, D]
          mllm::Tensor::shape_t,  // K shape [B, H_kv, S_kv, D]
          mllm::Tensor::shape_t,  // V shape [B, H_kv, S_kv, D]
          bool                     // use_causal_mask
      >>& test_cases) {
    using namespace mllm;  // NOLINT

    for (const auto& [q_shape, k_shape, v_shape, use_mask] : test_cases) {
      // Validate shapes
      MLLM_RT_ASSERT_EQ(q_shape.size(), 4);
      MLLM_RT_ASSERT_EQ(k_shape.size(), 4);
      MLLM_RT_ASSERT_EQ(v_shape.size(), 4);
      MLLM_RT_ASSERT_EQ(q_shape[0], k_shape[0]);  // Same batch size
      MLLM_RT_ASSERT_EQ(q_shape[0], v_shape[0]);
      MLLM_RT_ASSERT_EQ(k_shape[1], v_shape[1]);  // KV have same num_heads
      MLLM_RT_ASSERT_EQ(q_shape[3], k_shape[3]);  // Same head_dim
      MLLM_RT_ASSERT_EQ(k_shape[2], v_shape[2]);  // K and V have same sequence length

      int32_t B = static_cast<int32_t>(q_shape[0]);
      int32_t H_q = static_cast<int32_t>(q_shape[1]);   // num query heads
      int32_t H_kv = static_cast<int32_t>(k_shape[1]);  // num KV heads
      int32_t S_q = static_cast<int32_t>(q_shape[2]);
      int32_t S_kv = static_cast<int32_t>(k_shape[2]);
      int32_t D = static_cast<int32_t>(q_shape[3]);

      MLLM_RT_ASSERT_EQ(H_q % H_kv, 0);
      int32_t num_groups = H_q / H_kv;

      // 1. Create random FP16 inputs on CPU
      Tensor Q_cpu = Tensor::random(q_shape, -0.5f, 0.5f, kFloat16, kCPU);
      Tensor K_cpu = Tensor::random(k_shape, -0.5f, 0.5f, kFloat16, kCPU);
      Tensor V_cpu = Tensor::random(v_shape, -0.5f, 0.5f, kFloat16, kCPU);

      // 2. Create causal mask if needed
      Tensor mask_cpu;
      if (use_mask) {
        mask_cpu = Tensor::zeros({1, 1, S_q, S_kv}, kFloat16, kCPU);
        auto* mask_ptr = mask_cpu.ptr<mllm_fp16_t>();
        int32_t offset = S_kv - S_q;
        for (int32_t i = 0; i < S_q; ++i) {
          for (int32_t j = 0; j < S_kv; ++j) {
            if (j > i + offset) {
              mask_ptr[i * S_kv + j] = MLLM_FP32_TO_FP16(-10000.0f);
            }
          }
        }
      }

      // 3. Compute reference on CPU
      Tensor ref_cpu = computeGQACPU(Q_cpu, K_cpu, V_cpu, mask_cpu, use_mask, num_groups);

      // 4. Compute on Ascend
      auto Q_ascend = Q_cpu.to(kAscend);
      auto K_cpu_expanded = repeatKVHeads(K_cpu, num_groups);
      auto V_cpu_expanded = repeatKVHeads(V_cpu, num_groups);
      auto K_ascend = K_cpu_expanded.to(kAscend);
      auto V_ascend = V_cpu_expanded.to(kAscend);

      float scale = 1.0f / std::sqrt(static_cast<float>(D));

      // Q @ K^T
      auto scores = mllm::nn::functional::matmul(Q_ascend, K_ascend, false, true);

      // Scale
      auto scale_tensor_cpu = Tensor::ones({1}, kFloat16, kCPU);
      {
        auto* scale_ptr = scale_tensor_cpu.ptr<mllm_fp16_t>();
        scale_ptr[0] = MLLM_FP32_TO_FP16(scale);
      }
      auto scaled_scores = scores * scale_tensor_cpu.to(kAscend);

      // Add mask
      if (use_mask) {
        scaled_scores = scaled_scores + mask_cpu.to(kAscend);
      }

      // Softmax
      auto attn_weights = mllm::nn::functional::softmax(scaled_scores, -1);

      // attn_weights @ V
      auto output_ascend = mllm::nn::functional::matmul(attn_weights, V_ascend, false, false);

      // 5. Compare
      auto output_cpu = output_ascend.to(kCPU);
      auto result = mllm::test::allClose(output_cpu, ref_cpu, 5e-2f, 5e-2f);
      if (!result.is_close) {
        MLLM_ERROR("GQA test failed: B={}, H_q={}, H_kv={}, S_q={}, S_kv={}, D={}, mask={}",
                   B, H_q, H_kv, S_q, S_kv, D, use_mask ? "true" : "false");
        MLLM_ERROR("Max absolute diff: {}, Max relative diff: {}",
                   result.max_absolute_diff, result.max_relative_diff);
        return false;
      }

      MLLM_INFO("GQA test passed: B={}, H_q={}, H_kv={}, S_q={}, S_kv={}, D={}, mask={}",
                B, H_q, H_kv, S_q, S_kv, D, use_mask ? "true" : "false");
    }
    return true;
  }

 private:
  //===----------------------------------------------------------------------===//
  // Helper: Compute Multi-Head Attention reference on CPU (FP32)
  //===----------------------------------------------------------------------===//
  mllm::Tensor computeMultiHeadAttentionCPU(
      const mllm::Tensor& Q_cpu,
      const mllm::Tensor& K_cpu,
      const mllm::Tensor& V_cpu,
      const mllm::Tensor& mask_cpu,
      bool use_mask) {
    using namespace mllm;  // NOLINT

    int32_t B = static_cast<int32_t>(Q_cpu.shape()[0]);
    int32_t H = static_cast<int32_t>(Q_cpu.shape()[1]);
    int32_t S_q = static_cast<int32_t>(Q_cpu.shape()[2]);
    int32_t S_kv = static_cast<int32_t>(K_cpu.shape()[2]);
    int32_t D = static_cast<int32_t>(Q_cpu.shape()[3]);

    // Convert inputs to FP32
    Tensor Q_fp32 = Tensor::zeros({B, H, S_q, D}, kFloat32, kCPU);
    Tensor K_fp32 = Tensor::zeros({B, H, S_kv, D}, kFloat32, kCPU);
    Tensor V_fp32 = Tensor::zeros({B, H, S_kv, D}, kFloat32, kCPU);

    auto* q_fp16 = Q_cpu.ptr<mllm_fp16_t>();
    auto* k_fp16 = K_cpu.ptr<mllm_fp16_t>();
    auto* v_fp16 = V_cpu.ptr<mllm_fp16_t>();
    auto* q_fp32 = Q_fp32.ptr<mllm_fp32_t>();
    auto* k_fp32 = K_fp32.ptr<mllm_fp32_t>();
    auto* v_fp32 = V_fp32.ptr<mllm_fp32_t>();

    for (size_t i = 0; i < Q_cpu.numel(); ++i) {
      q_fp32[i] = MLLM_FP16_TO_FP32(q_fp16[i]);
    }
    for (size_t i = 0; i < K_cpu.numel(); ++i) {
      k_fp32[i] = MLLM_FP16_TO_FP32(k_fp16[i]);
    }
    for (size_t i = 0; i < V_cpu.numel(); ++i) {
      v_fp32[i] = MLLM_FP16_TO_FP32(v_fp16[i]);
    }

    // Convert mask to FP32 if needed
    const mllm_fp16_t* mask_fp16 = nullptr;
    if (use_mask) {
      mask_fp16 = mask_cpu.ptr<mllm_fp16_t>();
    }

    Tensor output_fp32 = Tensor::zeros({B, H, S_q, D}, kFloat32, kCPU);
    auto* out_ptr = output_fp32.ptr<mllm_fp32_t>();

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int32_t b = 0; b < B; ++b) {
      for (int32_t h = 0; h < H; ++h) {
        std::vector<float> scores(S_q * S_kv, 0.0f);
        for (int32_t i = 0; i < S_q; ++i) {
          for (int32_t j = 0; j < S_kv; ++j) {
            float sum = 0.0f;
            for (int32_t k = 0; k < D; ++k) {
              float q_val = q_fp32[((b * H + h) * S_q + i) * D + k];
              float k_val = k_fp32[((b * H + h) * S_kv + j) * D + k];
              sum += q_val * k_val;
            }
            scores[i * S_kv + j] = sum * scale;

            // Add mask (mask is broadcastable: [1, 1, S_q, S_kv])
            if (use_mask) {
              float mask_val = MLLM_FP16_TO_FP32(mask_fp16[i * S_kv + j]);
              scores[i * S_kv + j] += mask_val;
            }
          }
        }

        // Softmax along last dimension
        std::vector<float> attn_weights(S_q * S_kv);
        for (int32_t i = 0; i < S_q; ++i) {
          float max_val = -std::numeric_limits<float>::infinity();
          for (int32_t j = 0; j < S_kv; ++j) {
            max_val = std::max(max_val, scores[i * S_kv + j]);
          }

          float sum_exp = 0.0f;
          for (int32_t j = 0; j < S_kv; ++j) {
            float exp_val = std::exp(scores[i * S_kv + j] - max_val);
            attn_weights[i * S_kv + j] = exp_val;
            sum_exp += exp_val;
          }

          for (int32_t j = 0; j < S_kv; ++j) {
            attn_weights[i * S_kv + j] /= sum_exp;
          }
        }

        // Compute output: attn_weights @ V
        for (int32_t i = 0; i < S_q; ++i) {
          for (int32_t k = 0; k < D; ++k) {
            float sum = 0.0f;
            for (int32_t j = 0; j < S_kv; ++j) {
              float attn_val = attn_weights[i * S_kv + j];
              float v_val = v_fp32[((b * H + h) * S_kv + j) * D + k];
              sum += attn_val * v_val;
            }
            out_ptr[((b * H + h) * S_q + i) * D + k] = sum;
          }
        }
      }
    }

    // Convert output back to FP16
    Tensor output_fp16 = Tensor::zeros({B, H, S_q, D}, kFloat16, kCPU);
    auto* out_fp16 = output_fp16.ptr<mllm_fp16_t>();
    for (size_t i = 0; i < output_fp16.numel(); ++i) {
      out_fp16[i] = MLLM_FP32_TO_FP16(out_ptr[i]);
    }

    return output_fp16;
  }

  //===----------------------------------------------------------------------===//
  // Helper: Repeat KV heads for GQA
  // [B, H_kv, S, D] -> [B, H_q, S, D] where H_q = H_kv * num_groups
  //===----------------------------------------------------------------------===//
  mllm::Tensor repeatKVHeads(const mllm::Tensor& kv, int32_t num_groups) {
    using namespace mllm;  // NOLINT

    if (num_groups == 1) {
      return kv;
    }

    int32_t B = static_cast<int32_t>(kv.shape()[0]);
    int32_t H_kv = static_cast<int32_t>(kv.shape()[1]);
    int32_t S = static_cast<int32_t>(kv.shape()[2]);
    int32_t D = static_cast<int32_t>(kv.shape()[3]);
    int32_t H_q = H_kv * num_groups;

    Tensor expanded = Tensor::zeros({B, H_q, S, D}, kv.dtype(), kCPU);
    auto* src = kv.ptr<mllm_fp16_t>();
    auto* dst = expanded.ptr<mllm_fp16_t>();

    for (int32_t b = 0; b < B; ++b) {
      for (int32_t h_kv = 0; h_kv < H_kv; ++h_kv) {
        for (int32_t g = 0; g < num_groups; ++g) {
          int32_t h_q = h_kv * num_groups + g;
          for (int32_t s = 0; s < S; ++s) {
            for (int32_t d = 0; d < D; ++d) {
              size_t src_idx = ((b * H_kv + h_kv) * S + s) * D + d;
              size_t dst_idx = ((b * H_q + h_q) * S + s) * D + d;
              dst[dst_idx] = src[src_idx];
            }
          }
        }
      }
    }

    return expanded;
  }

  //===----------------------------------------------------------------------===//
  // Helper: Compute GQA reference on CPU
  //===----------------------------------------------------------------------===//
  mllm::Tensor computeGQACPU(
      const mllm::Tensor& Q_cpu,
      const mllm::Tensor& K_cpu,
      const mllm::Tensor& V_cpu,
      const mllm::Tensor& mask_cpu,
      bool use_mask,
      int32_t num_groups) {
    // Expand KV heads and compute standard MHA
    auto K_expanded = repeatKVHeads(K_cpu, num_groups);
    auto V_expanded = repeatKVHeads(V_cpu, num_groups);
    return computeMultiHeadAttentionCPU(Q_cpu, K_expanded, V_expanded, mask_cpu, use_mask);
  }
};
