// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/Parallel.hpp"
#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/backends/cpu/kernels/common/llamafile/llamafile_sgemm.hpp"

namespace mllm::cpu {

CPULinearOp::CPULinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void CPULinearOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      switch (options_.impl_type) {
        case aops::LinearImplTypes::kBLAS:
        case aops::LinearImplTypes::kDefault: {
          weight_ = weight_.view({options_.out_channels, options_.in_channels});
          if (options_.bias) {
            bias_ = ploader->pull(getName() + ".bias");
            bias_ = bias_.view({options_.out_channels});
          }
          break;
        }
        default: {
          // No need to view.
          MLLM_EMPTY_SCOPE
          break;
        }
      }
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      switch (options_.impl_type) {
        case aops::LinearImplTypes::kBLAS:
        case aops::LinearImplTypes::kDefault: {
          if (options_.bias) {
            bias_ = ploader->pull(getName() + ".bias");
            bias_ = bias_.view({options_.out_channels});
          }
          break;
        }
        default: {
          // No need to view.
          MLLM_EMPTY_SCOPE
          break;
        }
      }
      break;
    }
    default: NYI("Unsupported model file version")
  }
}

void CPULinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& o = outputs[0];

  auto impl_type = options_.impl_type;
  if (impl_type == aops::LinearImplTypes::kDefault) {
#if defined(MLLM_USE_BLAS)
    impl_type = aops::LinearImplTypes::kBLAS;
#endif
  }

  auto input_shape = input.shape();
  MLLM_RT_ASSERT(input_shape.size() >= 2);

  // In Linear
  // inputs is always: [..., S, in_channels]
  // outputs is always: [out_channels, in_channels]
  int M = input_shape[input_shape.size() - 2];
  int K = input_shape[input_shape.size() - 1];
  int N = options_.out_channels;
  MLLM_RT_ASSERT_EQ(K, options_.in_channels);

  int batch_count = 1;
  for (size_t i = 0; i < input_shape.size() - 2; ++i) { batch_count *= input_shape[i]; }

  switch (impl_type) {
    case aops::LinearImplTypes::kBLAS: {
#if defined(MLLM_USE_BLAS)
      MLLM_RT_ASSERT_EQ(input.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(weight_.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(o.dtype(), kFloat32);
      if (bias_) { MLLM_RT_ASSERT_EQ(bias_.dtype(), kFloat32); }
      if (batch_count == 1) {
        blas::matmul_fp32(input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(),
                          bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr, M, N, K, false, true);
      } else {
        blas::batch_matmul_fp32(input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(),
                                bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr, batch_count, M, N, K,
                                input.stride()[input_shape.size() - 3], 0, o.stride()[o.shape().size() - 3], false, true);
      }
#else
      NYI("BLAS not supported. Pls set MLLM_USE_BLAS=ON to enable BLAS supports in cmake.");
#endif
      break;
    }

// The code below is for ARM64/ARM.
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4);
      auto workspace = Tensor::empty({work_space_size}, kInt8, kCPU).alloc();

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4,
                        options_.getThreads());
      return;
    }
#endif
    case aops::LinearImplTypes::kDefault: {
      // Using llamafile
      // Use llamafile implementation for default case
      // Linear operation: output = input * weight^T + bias
      // In llamafile convention: C = A * B^T, where A is input, B is weight
      if (input.isContiguousN(0) && weight_.isContiguousN(0)) {
        auto thread_count = options_.getThreads();

        const int ld_input = K;   // Leading dimension of input
        const int ld_weight = K;  // Leading dimension of weight (weight is stored as [out_channels, in_channels])
        const int ld_output = N;  // Leading dimension of output

        void* bias_ptr = bias_ ? bias_.ptr<void>() : nullptr;
        DataTypes bias_type = bias_ ? bias_.dtype() : kFloat32;

        if (batch_count > 1) {
          // Handle batched linear operation
          for (int b = 0; b < batch_count; ++b) {
            MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, id, 0, thread_count, 1, {
              auto input_offset = input.stride()[input_shape.size() - 3] * b;
              auto output_offset = o.stride()[o.shape().size() - 3] * b;

              if (!llamafile_sgemm(N, M, K, weight_.ptr<mllm_byte_t>(), ld_weight,  // B matrix (weight)
                                   input.ptr<mllm_byte_t>() + input_offset * bytesOfType(input.dtype()),
                                   ld_input,  // A matrix (input)
                                   o.ptr<mllm_byte_t>() + output_offset * bytesOfType(o.dtype()), ld_output, id, thread_count,
                                   weight_.dtype(), input.dtype(), o.dtype(), bias_ptr, bias_type)) {
                MLLM_WARN("LlamaFile linear failed");
              }
            });
          }
        } else {
          // Single batch case
          MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, id, 0, thread_count, 1, {
            if (!llamafile_sgemm(N, M, K, weight_.ptr<mllm_byte_t>(), ld_weight,  // B matrix (weight)
                                 input.ptr<mllm_byte_t>(), ld_input,              // A matrix (input)
                                 o.ptr<mllm_byte_t>(), ld_output, id, thread_count, weight_.dtype(), input.dtype(), o.dtype(),
                                 bias_ptr, bias_type)) {
              MLLM_WARN("LlamaFile linear failed");
            }
          });
        }
      }
      break;
    }
    default: {
      NYI("LinearImplTypes not supported");
      break;
    }
  }
}

void CPULinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto i_shape = i.shape();

  MLLM_RT_ASSERT_EQ(i_shape[i_shape.size() - 1], options_.in_channels);

  auto o_shape = i_shape;
  o_shape[o_shape.size() - 1] = options_.out_channels;

  DataTypes o_dtype = i.dtype();

  switch (options_.impl_type) {
    case aops::LinearImplTypes::kKaiLinear_fp16_fp16_fp16p_mxk_kxn:
    case aops::LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4:
    case aops::LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4:
    case aops::LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm: {
      o_dtype = kFloat16;
      break;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4: {
      o_dtype = kFloat32;
      break;
    }
    default: o_dtype = i.dtype();
  }

  outputs.emplace_back(Tensor::empty(o_shape, o_dtype, i.device()));
}

}  // namespace mllm::cpu
