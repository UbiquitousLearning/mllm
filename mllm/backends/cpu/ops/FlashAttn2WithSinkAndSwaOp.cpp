// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/FlashAttn2WithSinkAndSwaOp.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1_swa_sink/fa2_swa_sink_fwd_bshd.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPUFlashAttention2SwaSinkOp::CPUFlashAttention2SwaSinkOp(const aops::FlashAttention2SwaSinkOptions& options)
    : aops::FlashAttention2SwaSinkOp(options) {}

void CPUFlashAttention2SwaSinkOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& S_AUX = inputs[3];
  auto& O = outputs[0];

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(Q.isContiguous());

  // NOTE:
  //
  // K, V is in flexible layout, no contiguous is OK.
  //
  // MLLM_RT_ASSERT(K.isContiguous());
  // MLLM_RT_ASSERT(V.isContiguous());

  auto B = Q.shape()[0];
  auto S_Q = Q.shape()[1];
  auto H_Q = Q.shape()[2];
  auto D_QK = Q.shape()[3];
  auto D_V = V.shape()[3];
  MLLM_RT_ASSERT_EQ(H_Q, options_.q_head);
  auto S_KV = K.shape()[1];
  MLLM_RT_ASSERT_EQ(S_KV, V.shape()[1]);

  switch (Q.dtype()) {
    case mllm::kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH)
      fwd_bshd_swa_with_sink<mllm::cpu::flash_attn2::details::__ArmArchTag, mllm::mllm_fp32_t, mllm::mllm_fp32_t,
                             mllm::mllm_fp32_t, mllm::mllm_fp32_t, mllm::mllm_fp32_t>(
          B, options_.q_head, options_.kv_head, S_Q, S_KV, D_QK, D_V, options_.sliding_window, options_.cur_seq_len,
          Q.ptr<mllm::mllm_fp32_t>(), K.ptr<mllm::mllm_fp32_t>(), V.ptr<mllm::mllm_fp32_t>(), S_AUX.ptr<mllm::mllm_fp32_t>(),
          O.ptr<mllm::mllm_fp32_t>(), options_.getThreads());
#elif defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)
      fwd_bshd_swa_with_sink<mllm::cpu::flash_attn2::details::__X86ArchTag, mllm::mllm_fp32_t, mllm::mllm_fp32_t,
                             mllm::mllm_fp32_t, mllm::mllm_fp32_t, mllm::mllm_fp32_t>(
          B, options_.q_head, options_.kv_head, S_Q, S_KV, D_QK, D_V, options_.sliding_window, options_.cur_seq_len,
          Q.ptr<mllm::mllm_fp32_t>(), K.ptr<mllm::mllm_fp32_t>(), V.ptr<mllm::mllm_fp32_t>(), S_AUX.ptr<mllm::mllm_fp32_t>(),
          O.ptr<mllm::mllm_fp32_t>(), options_.getThreads());
#endif
      break;
    }
    default: NYI("FlashAttention2SwaSink::forward not support dtype {}", nameOfType(Q.dtype())); break;
  }
}

}  // namespace mllm::cpu
