// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/utils/Common.hpp"

#include "KernelTestHelper.hpp"

class LlamaFileKernelTest : public KernelTest {
 public:
  LlamaFileKernelTest() = default;
  ~LlamaFileKernelTest() override = default;

  mllm::Tensor matmulReference(const mllm::Tensor& A, const mllm::Tensor& B, bool transpose_a = false,
                               bool transpose_b = false) {
    using mllm::ExitCode;
    using mllm::kCPU;
    using mllm::kFloat32;
    using mllm::Tensor;

    auto a_shape = A.shape();
    auto b_shape = B.shape();

    MLLM_RT_ASSERT(a_shape.size() == b_shape.size());

    if (a_shape.size() == 2 && b_shape.size() == 2) {
      int M, K, N, K_other;
      if (!transpose_a) {
        M = a_shape[a_shape.size() - 2];
        K = a_shape[a_shape.size() - 1];
      } else {
        K = a_shape[a_shape.size() - 2];
        M = a_shape[a_shape.size() - 1];
      }

      if (!transpose_b) {
        K_other = b_shape[b_shape.size() - 2];
        N = b_shape[b_shape.size() - 1];
      } else {
        N = b_shape[b_shape.size() - 2];
        K_other = b_shape[b_shape.size() - 1];
      }

      MLLM_RT_ASSERT(K == K_other);

      auto ref_output = mllm::Tensor::empty({M, N}, kFloat32);
      ref_output.alloc();

      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
          float sum = 0.0f;
          for (int k = 0; k < K; k++) {
            float a_val = transpose_a ? A.constAt<float>({k, m}) : A.constAt<float>({m, k});
            float b_val = transpose_b ? B.constAt<float>({n, k}) : B.constAt<float>({k, n});
            sum += a_val * b_val;
          }
          ref_output.at<float>({m, n}) = sum;
        }
      }
      return ref_output;
    } else if (a_shape.size() > 2 && b_shape.size() > 2) {
      // batch matmul
      int batch_count = 1;
      for (size_t i = 0; i < a_shape.size() - 2; ++i) { batch_count *= a_shape[i]; }

      int M, K, N, K_other;
      if (!transpose_a) {
        M = a_shape[a_shape.size() - 2];
        K = a_shape[a_shape.size() - 1];
      } else {
        K = a_shape[a_shape.size() - 2];
        M = a_shape[a_shape.size() - 1];
      }

      if (!transpose_b) {
        K_other = b_shape[b_shape.size() - 2];
        N = b_shape[b_shape.size() - 1];
      } else {
        N = b_shape[b_shape.size() - 2];
        K_other = b_shape[b_shape.size() - 1];
      }

      MLLM_RT_ASSERT(K == K_other);

      auto ref_output = mllm::Tensor::empty({A.shape()[0], M, N}, kFloat32);
      ref_output.alloc();

      for (int i = 0; i < batch_count; ++i) {
        for (int m = 0; m < M; ++m) {
          for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
              float a_val = transpose_a ? A.constAt<float>({i, k, m}) : A.constAt<float>({i, m, k});
              float b_val = transpose_b ? B.constAt<float>({i, n, k}) : B.constAt<float>({i, k, n});
              sum += a_val * b_val;
            }
            ref_output.at<float>({i, m, n}) = sum;
          }
        }
      }

      return ref_output;
    } else {
      MLLM_WARN("Unsupported tensor dimensions in {}", __FUNCTION__);
      return Tensor{};
    }
  }

  bool oneCase(const std::pair<mllm::Tensor::shape_t, mllm::Tensor::shape_t>& shape, const bool transpose_a = false,
               const bool transpose_b = false) {
    auto a_shape = shape.first;
    auto b_shape = shape.second;

    auto A = mllm::Tensor::random(a_shape);
    auto B = mllm::Tensor::random(b_shape);

    auto result = mllm::nn::functional::matmul(A, B, transpose_a, transpose_b, mllm::aops::MatMulOpType::kGGUF);

    auto ref_output = matmulReference(A, B, transpose_a, transpose_b);

    auto compare_res = mllm::test::allClose(result, ref_output);

    if (!compare_res) { mllm::print(compare_res); }
    return compare_res.is_close;
  }
};
