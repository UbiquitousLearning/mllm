// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <unordered_map>

#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"

#include "KernelTestHelper.hpp"

class MllmBlasArmSgemmKernelTest : public KernelTest {
 public:
  MllmBlasArmSgemmKernelTest() = default;
  ~MllmBlasArmSgemmKernelTest() override = default;

  bool cmp_mllm_blas_matmul_fp32_gemm_nt_nt(std::unordered_map<std::string, int32_t>& vars) {
    int D = vars["D"];
    int S_Q = vars["S_Q"];
    int S_KV = vars["S_KV"];

    auto A = mllm::Tensor::random({S_Q, S_KV}, mllm::kFloat32, mllm::kCPU);
    auto B = mllm::Tensor::random({S_KV, D}, mllm::kFloat32, mllm::kCPU);

    auto RefDST = mllm::nn::functional::matmul(A, B, false, false, mllm::aops::MatMulOpType::kMllmBlas);
    auto DST = mllm::Tensor::emptyLike(RefDST).alloc();

    // Calculate DST.
    {
      auto dst_ptr = DST.ptr<float>();
      auto a_ptr = A.ptr<float>();
      auto b_ptr = B.ptr<float>();
      const int M = S_Q;
      const int K = S_KV;
      const int N = D;
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k) { sum += a_ptr[i * K + k] * b_ptr[k * N + j]; }
          dst_ptr[i * N + j] = sum;
        }
      }
    }

    auto result = mllm::test::allClose(DST, RefDST);
    if (!result.is_close) {
      mllm::print(result);
      mllm::print("D: ", D, ", S_Q: ", S_Q, ", S_KV: ", S_KV);
      return false;
    }
    return true;
  }

  bool test_mllm_blas_matmul_fp32_gemm_nt_nt(const std::vector<std::unordered_map<std::string, int32_t>>& vars) {
    for (auto v : vars) {
      if (!cmp_mllm_blas_matmul_fp32_gemm_nt_nt(v)) { return false; }
    }
    return true;
  }

  bool cmp_mllm_blas_matmul_fp32_gemm_nt_t(std::unordered_map<std::string, int32_t>& vars) {
    int in_channels = vars["in_channels"];
    int out_channels = vars["out_channels"];
    int batch = vars["batch"];

    auto A = mllm::Tensor::random({batch, in_channels}, mllm::kFloat32, mllm::kCPU);
    auto B = mllm::Tensor::random({out_channels, in_channels}, mllm::kFloat32, mllm::kCPU);

    auto RefDST = mllm::nn::functional::matmul(A, B, false, true, mllm::aops::MatMulOpType::kMllmBlas);
    auto DST = mllm::Tensor::emptyLike(RefDST).alloc();

    // Calculate DST.
    {
      auto dst_ptr = DST.ptr<float>();
      auto a_ptr = A.ptr<float>();
      auto b_ptr = B.ptr<float>();
      const int M = batch;
      const int K = in_channels;
      const int N = out_channels;
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k) { sum += a_ptr[i * K + k] * b_ptr[j * K + k]; }
          dst_ptr[i * N + j] = sum;
        }
      }
    }

    auto result = mllm::test::allClose(DST, RefDST);
    if (!result.is_close) {
      mllm::print(result);
      mllm::print("in_channel: ", in_channels, ", out_channels: ", out_channels, ", batch: ", batch);
      return false;
    }
    return true;
  }

  bool test_mllm_blas_matmul_fp32_gemm_nt_t(const std::vector<std::unordered_map<std::string, int32_t>>& vars) {
    for (auto v : vars) {
      if (!cmp_mllm_blas_matmul_fp32_gemm_nt_t(v)) { return false; }
    }
    return true;
  }
};
