// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/Tensor.hpp"
#include "mllm/backends/cpu/ops/RadixAttnWithSinkAndSwaDiffDimOp.hpp"
#include "mllm/backends/cpu/kernels/common/radix_swa_sink/radix_swa_sink_fwd_bshd.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPURadixAttnSwaSinkOp::CPURadixAttnSwaSinkOp(const aops::RadixAttnSwaSinkOptions& options)
    : aops::RadixAttnSwaSinkOp(options) {}

void CPURadixAttnSwaSinkOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
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
  MLLM_RT_ASSERT_EQ(H_Q, options_.q_head);
  auto S_KV = K.shape()[0];
  MLLM_RT_ASSERT_EQ(S_KV, V.shape()[0]);

  switch (Q.dtype()) {
    case mllm::kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      switch (options_.pattern) {
        case aops::RadixAttnSwaSinkPattern::kDecode: {
          fwd_bshd_decode<::mllm::cpu::radix_attn::details::__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                          mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V,
                                       options_.sliding_window, options_.cur_seq_len, Q.ptr<mllm_fp32_t>(),
                                       K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), S_AUX.ptr<mllm_fp32_t>(),
                                       O.ptr<mllm_fp32_t>(), options_.getThreads());
          break;
        }
        case aops::RadixAttnSwaSinkPattern::kPrefill: {
          fwd_bshd_prefill<::mllm::cpu::radix_attn::details::__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                           mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V,
                                        options_.sliding_window, options_.cur_seq_len, Q.ptr<mllm_fp32_t>(),
                                        K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), S_AUX.ptr<mllm_fp32_t>(),
                                        O.ptr<mllm_fp32_t>(), options_.getThreads());
          break;
        }
        case aops::RadixAttnSwaSinkPattern::kAppend: {
          // TODO
          break;
        }
      }
#elif defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)
      switch (options_.pattern) {
        case aops::RadixAttnSwaSinkPattern::kDecode: {
          fwd_bshd_decode<::mllm::cpu::radix_attn::details::__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                          mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V,
                                       options_.sliding_window, options_.cur_seq_len, Q.ptr<mllm_fp32_t>(),
                                       K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), S_AUX.ptr<mllm_fp32_t>(),
                                       O.ptr<mllm_fp32_t>(), options_.getThreads());
          break;
        }
        case aops::RadixAttnSwaSinkPattern::kPrefill: {
          fwd_bshd_prefill<::mllm::cpu::radix_attn::details::__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                           mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V,
                                        options_.sliding_window, options_.cur_seq_len, Q.ptr<mllm_fp32_t>(),
                                        K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), S_AUX.ptr<mllm_fp32_t>(),
                                        O.ptr<mllm_fp32_t>(), options_.getThreads());
          break;
        }
        case aops::RadixAttnSwaSinkPattern::kAppend: {
          // TODO
          break;
        }
      }
#endif
      break;
    }
    default: NYI("RadixAttnSwaSink::forward not support dtype {}", nameOfType(Q.dtype())); break;
  }
}

}  // namespace mllm::cpu
