// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/FlashAttention2Op.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/fwd_bshd.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPUFlashAttention2Op::CPUFlashAttention2Op(const aops::FlashAttention2OpOptions& options) : aops::FlashAttention2Op(options) {}

void CPUFlashAttention2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[0];
  auto& V = inputs[0];
  auto& Y = outputs[0];

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(Q.isContiguous());
  MLLM_RT_ASSERT(K.isContiguous());
  MLLM_RT_ASSERT(V.isContiguous());

  switch (Q.dtype()) {
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      using fa2_impl_arch = fa2::NativeArchTrait;
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      using fa2_impl_arch = fa2::ArmNeon128ArchTrait;
#endif

      // Only Support BSHD input

      auto query_seq_len = Q.shape()[1];
      auto key_seq_len = K.shape()[1];

      using fa2_fp16_cfg = fa2::FlashAttention2Config<
          fa2::KernelConfig<fa2_impl_arch, fa2::DefaultMma0Layout, fa2::DefaultMma1Layout,
                            fa2::NumericTrait<mllm_fp32_t, mllm_fp16_t>, fa2::MemoryTrait<128>, fa2::TileTrait<4, 4, -1>>,
          /*causal_mask*/ true, /*sliding_window*/ false,
          /*dropout*/ false, /*high precision exp*/ false>;

      // FIXME Use global tensor
      // clang-format off
      auto acc_s_cast = Tensor::empty({options_.getThreads(), fa2_fp16_cfg::Tile::kTileM, fa2_fp16_cfg::Tile::kTileN}, kFloat16, kCPU).alloc();
      auto acc_o = Tensor::empty({options_.getThreads(), fa2_fp16_cfg::Tile::kTileM, options_.D}, kFloat32, kCPU).alloc();
      auto acc_s = Tensor::empty({options_.getThreads(), fa2_fp16_cfg::Tile::kTileM, fa2_fp16_cfg::Tile::kTileN}, kFloat32, kCPU).alloc();
      auto logsum = Tensor::empty({options_.getThreads(), 4}, kFloat32, kCPU).alloc();
      auto scoremax = Tensor::empty({options_.getThreads(), 4}, kFloat32, kCPU).alloc();
      auto scoremax_prev = Tensor::empty({options_.getThreads(), 4}, kFloat32, kCPU).alloc();
      auto score_scale = Tensor::empty({options_.getThreads(), 4}, kFloat32, kCPU).alloc();
      auto score_sum = Tensor::empty({options_.getThreads(), 4}, kFloat32, kCPU).alloc();
      // clang-format on

      fa2::FlashAttention2WorkSpace<fa2_fp16_cfg> workspace{
          .acc_s_cast = acc_s_cast.ptr<fa2_fp16_cfg::Numeric::ElementCompute>(),
          .acc_o = acc_o.ptr<fa2_fp16_cfg::Numeric::ElementAccumulator>(),
          .acc_s = acc_s.ptr<fa2_fp16_cfg::Numeric::ElementAccumulator>(),
          .logsum = logsum.ptr<fa2_fp16_cfg::Numeric::ElementAccumulator>(),
          .scoremax = scoremax.ptr<fa2_fp16_cfg::Numeric::ElementAccumulator>(),
          .scoremax_prev = scoremax_prev.ptr<fa2_fp16_cfg::Numeric::ElementAccumulator>(),
          .score_scale = score_scale.ptr<fa2_fp16_cfg::Numeric::ElementAccumulator>(),
          .score_sum = score_sum.ptr<fa2_fp16_cfg::Numeric::ElementAccumulator>(),
      };

      fa2::FlashAttention2<fa2_fp16_cfg> fa2_op;
      fa2_op.run(workspace, Q.ptr<fa2_fp16_cfg::Numeric::ElementCompute>(), K.ptr<fa2_fp16_cfg::Numeric::ElementCompute>(),
                 V.ptr<fa2_fp16_cfg::Numeric::ElementCompute>(), Y.ptr<fa2_fp16_cfg::Numeric::ElementCompute>(), options_.B,
                 options_.q_head, options_.kv_head, query_seq_len, key_seq_len, options_.D, options_.getThreads());
      break;
    }
    default: NYI("CPUQuickFlashAttention2Op::forward not support dtype {}", nameOfType(Q.dtype())); break;
  }
}

}  // namespace mllm::cpu
