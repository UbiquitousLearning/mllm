// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>

#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"
#include "mllm/nn/Functional.hpp"

class BlasKernelTest : public KernelTest {
 public:
  BlasKernelTest() = default;
  ~BlasKernelTest() override = default;

  bool matmul_MxK_NxK(const std::vector<std::unordered_map<std::string, int32_t>>& vars) {
    for (auto v : vars) {
      int M = v["M"];
      int N = v["N"];
      int K = v["K"];
      auto A = mllm::Tensor::random({M, K});
      auto B = mllm::Tensor::random({N, K});
      auto C = mllm::nn::functional::matmul(A, B, false, true, mllm::aops::MatMulOpType::kBLAS);

      auto C_ref = mllm::Tensor::zeros({M, N});
      auto a_ptr = A.ptr<float>();
      auto b_ptr = B.ptr<float>();
      auto c_ref_ptr = C_ref.ptr<float>();

      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k) { sum += a_ptr[i * K + k] * b_ptr[j * K + k]; }
          c_ref_ptr[i * N + j] = sum;
        }
      }

      auto close = mllm::test::allClose(C, C_ref);
      if (!close) {
        mllm::print(close);
        return false;
      }
    }
    return true;
  }

  bool batch_matmul_BHSD(const std::vector<std::unordered_map<std::string, int32_t>>& vars) {
    for (auto v : vars) {
      int Batch = v["B"];
      int H = v["H"];
      int S = v["S"];
      int D = v["D"];
      auto A = mllm::Tensor::random({Batch, H, S, D});
      auto B = mllm::Tensor::random({Batch, H, S, D});
      auto C = mllm::nn::functional::matmul(A, B, false, true, mllm::aops::MatMulOpType::kBLAS);

      auto C_ref = mllm::Tensor::zeros({Batch, H, S, S});
      auto a_ptr = A.ptr<float>();
      auto b_ptr = B.ptr<float>();
      auto c_ref_ptr = C_ref.ptr<float>();

      // Refcode from QWEN!
      for (int b = 0; b < Batch; ++b) {
        for (int h = 0; h < H; ++h) {
          // Pointer offset for the current batch and head
          int batch_head_offset_A = (b * H + h) * S * D;
          int batch_head_offset_B = (b * H + h) * S * D;
          int batch_head_offset_C = (b * H + h) * S * S;

          // Matrix multiplication for a single head in the batch
          // C[i,j] = sum over k of A[i,k] * B[j,k]
          for (int i = 0; i < S; ++i) {    // Row index for matrix A and C
            for (int j = 0; j < S; ++j) {  // Row index for matrix B (column index for B^T and C)
              float sum = 0.0f;
              for (int k = 0; k < D; ++k) {  // Column index for A, and row index for B^T
                // Index for A[b, h, i, k]
                int a_idx = batch_head_offset_A + i * D + k;
                // Index for B[b, h, j, k] (note j and k)
                int b_idx = batch_head_offset_B + j * D + k;
                sum += a_ptr[a_idx] * b_ptr[b_idx];
              }
              // Index for C[b, h, i, j]
              int c_idx = batch_head_offset_C + i * S + j;
              c_ref_ptr[c_idx] = sum;
            }
          }
        }
      }

      auto close = mllm::test::allClose(C, C_ref);
      if (!close) {
        mllm::print(close);
        return false;
      }
    }
    return true;
  }
};
