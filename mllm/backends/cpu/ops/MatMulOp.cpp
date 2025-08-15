// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/MatMulOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/backends/cpu/kernels/common/blas.hpp"

namespace mllm::cpu {

CPUMatMulOp::CPUMatMulOp(const aops::MatMulOpOptions& options) : aops::MatMulOp(options) {}

void CPUMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& lhs = inputs[0];
  auto& rhs = inputs[1];
  auto& o = outputs[0];

  auto transpose_a = options_.transpose_a;
  auto transpose_b = options_.transpose_b;

  auto mt = options_.matmul_type;
  if (mt == aops::MatMulOpType::kDefault) {
#if defined(MLLM_USE_BLAS)
    mt = aops::MatMulOpType::kBLAS;
#endif
  }

  auto lhs_shape = lhs.shape();
  auto rhs_shape = rhs.shape();

  MLLM_RT_ASSERT(lhs_shape.size() >= 2);
  MLLM_RT_ASSERT(rhs_shape.size() >= 2);

  const int lhs_rows = lhs_shape[lhs_shape.size() - 2];
  const int lhs_cols = lhs_shape[lhs_shape.size() - 1];
  const int rhs_rows = rhs_shape[rhs_shape.size() - 2];
  const int rhs_cols = rhs_shape[rhs_shape.size() - 1];
  const int M = transpose_a ? lhs_cols : lhs_rows;
  const int N = transpose_b ? rhs_rows : rhs_cols;
  const int K = transpose_a ? lhs_rows : lhs_cols;
  const int K_from_rhs = transpose_b ? rhs_cols : rhs_rows;
  MLLM_RT_ASSERT_EQ(K, K_from_rhs);

  int batch_count = 1;
  for (size_t i = 0; i < lhs_shape.size() - 2; ++i) { batch_count *= lhs_shape[i]; }
  int rhs_batch_count = 1;
  for (size_t i = 0; i < rhs_shape.size() - 2; ++i) { rhs_batch_count *= rhs_shape[i]; }
  MLLM_RT_ASSERT_EQ(batch_count, rhs_batch_count);

  switch (mt) {
    case aops::MatMulOpType::kBLAS: {
#if defined(MLLM_USE_BLAS)
      MLLM_RT_ASSERT_EQ(lhs.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(rhs.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(o.dtype(), kFloat32);
      if (batch_count == 1) {
        blas::matmul_fp32(lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(), nullptr, M, N, K, transpose_a,
                          transpose_b);
      } else {
        blas::batch_matmul_fp32(lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(), nullptr, batch_count, M,
                                N, K, lhs.stride()[lhs_shape.size() - 3], rhs.stride()[rhs_shape.size() - 3],
                                o.stride()[o.shape().size() - 3], transpose_a, transpose_b);
      }
#else
      NYI("BLAS not supported. Pls set MLLM_USE_BLAS=ON to enable BLAS supports in cmake.");
#endif
      break;
    }
    case aops::MatMulOpType::kDefault: {
      break;
    }
    default: {
      NYI("MatMulOpType not supported");
      break;
    }
  }
}

}  // namespace mllm::cpu
