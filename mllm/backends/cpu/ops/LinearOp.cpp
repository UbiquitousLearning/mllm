// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::cpu {

CPULinearOp::CPULinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void CPULinearOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      switch (options_.impl_type) {
        case aops::LinearImplTypes::kBLAS:
        case aops::LinearImplTypes::kGGUF:
        case aops::LinearImplTypes::kMllmBlas:
        case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_NT_NEON:
        case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_T_SME:
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
        case aops::LinearImplTypes::kGGUF:
        case aops::LinearImplTypes::kMllmBlas:
        case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_NT_NEON:
        case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_T_SME:
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

  // Prepare data:
  auto impl_type = options_.impl_type;
  if (impl_type == aops::LinearImplTypes::kDefault) {
#if defined(MLLM_USE_BLAS)
    impl_type = aops::LinearImplTypes::kBLAS;
#else
    // FIXME, When we need kMllmBlas_KAI_SGEMM_NT_NT_NEON. set it.
#endif
  }

  switch (impl_type) {
    case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_NT_NEON: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      ::mllm::cpu::arm::KaiLinear_fp32_fp32_fp32p_mxk_kxn kai_helper;
      weight_ = weight_.view({options_.out_channels, options_.in_channels});
      auto transposed_weight = weight_.transpose(0, 1);
      int32_t packed_weight_size = kai_helper.quant_pack_rhs_size(transposed_weight.size(0), transposed_weight.size(1));
      auto packed_weight = Tensor::empty({packed_weight_size}, kInt8, kCPU).alloc().setName(weight_.name()).setMemType(kGlobal);
      kai_helper.quant_pack_rhs_offline(packed_weight.ptr<mllm_byte_t>(), transposed_weight.ptr<mllm_fp32_t>(),
                                        bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr, transposed_weight.size(0),
                                        transposed_weight.size(1));
      MLLM_INFO("Packing fp32 weight and bias to kai's fp32 format");
      weight_ = packed_weight;
#endif
      break;
    }
    default: {
      // No need to postprocess.
      MLLM_EMPTY_SCOPE
      break;
    }
  }
}

void CPULinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& o = outputs[0];

  auto input_shape = input.shape();
  MLLM_RT_ASSERT(input_shape.size() >= 2);

  // In Linear
  // inputs is always: [..., S, in_channels]
  // outputs is always: [out_channels, in_channels]
  int M = input_shape[input_shape.size() - 2];
  int K = input_shape[input_shape.size() - 1];
  int N = options_.out_channels;
  MLLM_RT_ASSERT_EQ(K, options_.in_channels);

  auto impl_type = options_.impl_type;
  if (impl_type == aops::LinearImplTypes::kDefault) {
#if defined(MLLM_USE_BLAS)
    impl_type = aops::LinearImplTypes::kBLAS;
#else
    if (K >= 4) {
      impl_type = aops::LinearImplTypes::kGGUF;
    } else
    // All fallback to mllm blas
    {
      impl_type = aops::LinearImplTypes::kMllmBlas;
    }
#endif
  }

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
    case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_NT_NEON: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_fp32_fp32_fp32p_mxk_kxn kai_helper;
      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), nullptr, M, K, N,
                        options_.getThreads());

      break;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32: {
    __mllm_label_kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32:
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
    __mllm_label_kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32:
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

      if (M == 1) { goto __mllm_label_kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32; }

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

      if (M == 1) { goto __mllm_label_kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32; }

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
    case aops::LinearImplTypes::kGGUF: {
      // use ggml matmul, which first try llamafile_sgemm, then fallback to ggml matmul
      auto thread_count = options_.getThreads();
      auto* bias_ptr = options_.bias ? &bias_ : nullptr;
      mllm::cpu::ggml::mat_mul(input, weight_, o, options_.bias, bias_ptr, false, true, thread_count);
      break;
    }
    case aops::LinearImplTypes::kMllmBlas: {
      MLLM_RT_ASSERT_EQ(input.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(weight_.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(o.dtype(), kFloat32);
      if (bias_) { MLLM_RT_ASSERT_EQ(bias_.dtype(), kFloat32); }

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      if (batch_count == 1) {
        arm::mllm_blas_matmul_fp32(M, K, N, o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(),
                                   options_.bias ? bias_.ptr<mllm_fp32_t>() : nullptr, false, true, options_.getThreads());
      } else {
        arm::mllm_blas_batch_matmul_fp32(batch_count, M, K, N, o.stride()[o.shape().size() - 3],
                                         input.stride()[input.rank() - 3], 0, 0, o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(),
                                         weight_.ptr<mllm_fp32_t>(), options_.bias ? bias_.ptr<mllm_fp32_t>() : nullptr, false,
                                         true, options_.getThreads());
      }
#else
// TODO Other arch
#endif
      break;
    }
    default: {
      NYI("LinearImplTypes not supported");
      break;
    }
  }
}

void CPULinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isRedirect()) {
    outputs.emplace_back(inputs[1]);
    return;
  }
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
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm: {
      o_dtype = kFloat32;
      break;
    }
    case aops::LinearImplTypes::kGGUF: {
      o_dtype = kFloat32;
      break;
    }
    default: o_dtype = i.dtype();
  }

  outputs.emplace_back(Tensor::empty(o_shape, o_dtype, i.device()));
}

}  // namespace mllm::cpu
