// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/backends/cpu/ops/MatMulOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/backends/cpu/kernels/common/blas.hpp"
#include "mllm/backends/cpu/kernels/common/llamafile/llamafile_sgemm.hpp"

namespace mllm::cpu {

CPUMatMulOp::CPUMatMulOp(const aops::MatMulOpOptions& options) : aops::MatMulOp(options) {}

void CPUMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& lhs = inputs[0];
  auto& rhs = inputs[1];
  auto& o = outputs[0];

  auto transpose_a = options_.transpose_a;
  auto transpose_b = options_.transpose_b;

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

  auto mt = options_.matmul_type;
  if (mt == aops::MatMulOpType::kDefault) {
#if defined(MLLM_USE_BLAS)
    mt = aops::MatMulOpType::kBLAS;
#else
    if (!transpose_a && transpose_b && M != 1) {
      // FIXME: kLLamaFile is not supported for M == 1
      mt = aops::MatMulOpType::kLlamaFile;
    } else
    // All fallback to mllm blas
    {
      mt = aops::MatMulOpType::kMllmBlas;
    }
#endif
  }

  switch (mt) {
    case aops::MatMulOpType::kDefault: {
      NYI("Auto matmul type inference failed");
      break;
    }
    case aops::MatMulOpType::kLlamaFile: {
      // llamafile implementation
      // only supports specific transpose options
      MLLM_RT_ASSERT(transpose_a == false && transpose_b == true);

      // llamafile uses column-major order, so we actually perform K^T x Q
      if (lhs.isContiguousN(0)) {
        auto thread_count = options_.getThreads();

        auto src1_type_size = bytesOfType(rhs.dtype());
        auto src1_blck_size = lanesOfType(rhs.dtype());
        auto src0_type_size = bytesOfType(lhs.dtype());
        auto src0_blck_size = lanesOfType(lhs.dtype());

        const int ld0 = transpose_a ? M : K;
        const int ld1 = transpose_b ? K : N;
        const int ldc = o.shape()[o.shape().size() - 1];

        if (lhs_shape.size() > 2) {
          for (int b = 0; b < batch_count; ++b) {
            MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, id, 0, thread_count, 1, {
              auto offset = (rhs.stride()[rhs_shape.size() - 3] * b / src1_blck_size * src1_type_size);
              if (!llamafile_sgemm(
                      N, M, K / src1_blck_size,
                      rhs.ptr<mllm_byte_t>() + rhs.stride()[rhs_shape.size() - 3] * b / src1_blck_size * src1_type_size,
                      ld1 / src1_blck_size,
                      lhs.ptr<mllm_byte_t>() + lhs.stride()[lhs_shape.size() - 3] * b / src0_blck_size * src0_type_size,
                      ld0 / src0_blck_size,
                      o.ptr<mllm_byte_t>()
                          + o.stride()[o.shape().size() - 3] * b / lanesOfType(o.dtype()) * bytesOfType(o.dtype()),
                      ldc / lanesOfType(o.dtype()), id, thread_count, rhs.dtype(), lhs.dtype(), o.dtype())) {
                MLLM_WARN("LlamaFile matmul failed");
              }
            });
          }
        } else {
          MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, id, 0, thread_count, 1, {
            if (!llamafile_sgemm(N, M, K / src1_blck_size, rhs.ptr<mllm_byte_t>(), ld1 / src1_blck_size, lhs.ptr<mllm_byte_t>(),
                                 ld0 / src0_blck_size, o.ptr<mllm_byte_t>(), ldc / lanesOfType(o.dtype()), id, thread_count,
                                 rhs.dtype(), lhs.dtype(), o.dtype())) {
              MLLM_WARN("LlamaFile matmul failed");
            }
          });
        }
      } else {
        MLLM_WARN("LlamaFile matmul requires contiguous tensors, falling back to default implementation.");
        NYI("Default implementation for non-contiguous tensors not yet implemented");
      }
      break;
    }
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
    case aops::MatMulOpType::kMllmBlas: {
      auto thread_count = options_.getThreads();
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      if (lhs.dtype() == kFloat32 && rhs.dtype() == kFloat32 && o.dtype() == kFloat32) {
        if (batch_count == 1) {
          arm::mllm_blas_matmul_fp32(M, K, N, o.ptr<mllm_fp32_t>(), lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), nullptr,
                                     transpose_a, transpose_b, thread_count);
        } else {
          arm::mllm_blas_batch_matmul_fp32(batch_count, M, K, N, o.stride()[o.shape().size() - 3],
                                           lhs.stride()[lhs_shape.size() - 3], rhs.stride()[rhs_shape.size() - 3], 0,
                                           o.ptr<mllm_fp32_t>(), lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), nullptr,
                                           transpose_a, transpose_b, thread_count);
        }
      }
#else
      NYI("MllmBlas only support MLLM_HOST_ARCH_ARM64 or MLLM_HOST_ARCH_ARM right now.")
#endif
      break;
    }
    default: {
      break;
    }
  }
}

}  // namespace mllm::cpu
