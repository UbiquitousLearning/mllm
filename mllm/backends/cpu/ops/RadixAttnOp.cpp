// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/Tensor.hpp"
#include "mllm/backends/cpu/ops/RadixAttnOp.hpp"
#include "mllm/backends/cpu/kernels/common/radix_attn/arch.hpp"
#include "mllm/backends/cpu/kernels/common/radix_attn/fwd_bshd.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPURadixAttnOp::CPURadixAttnOp(const aops::RadixAttnOpOptions& options) : aops::RadixAttnOp(options) {}

void CPURadixAttnOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& Q = inputs[0];
  const auto& K_PTR = inputs[1];
  const auto& V_PTR = inputs[2];
  const auto& OUT = outputs[0];

  MLLM_RT_ASSERT(K_PTR.dtype() == kInt64 && V_PTR.dtype() == kInt64 && K_PTR.rank() == 1 && V_PTR.rank() == 1);
  auto B = Q.shape()[0];
  auto S_Q = Q.shape()[1];
  auto H_Q = Q.shape()[2];
  auto D = Q.shape()[3];
  MLLM_RT_ASSERT_EQ(H_Q, options_.H_Q);
  auto S_KV = K_PTR.shape()[0];
  MLLM_RT_ASSERT_EQ(S_KV, V_PTR.shape()[0]);

  switch (Q.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH)
      cpu::radix_attn::fwd_bshd<cpu::radix_attn::details::__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                                mllm_fp32_t>(B, options_.H_Q, options_.H_KV, S_Q, S_KV, D, Q.ptr<mllm_fp32_t>(),
                                             K_PTR.ptr<mllm_fp32_t*>(), V_PTR.ptr<mllm_fp32_t*>(), OUT.ptr<mllm_fp32_t>(),
                                             options_.getThreads());
#elif defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)
      cpu::radix_attn::fwd_bshd<cpu::radix_attn::details::__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                                mllm_fp32_t>(B, options_.H_Q, options_.H_KV, S_Q, S_KV, D, Q.ptr<mllm_fp32_t>(),
                                             K_PTR.ptr<mllm_fp32_t*>(), V_PTR.ptr<mllm_fp32_t*>(), OUT.ptr<mllm_fp32_t>(),
                                             options_.getThreads());
#endif
      break;
    }
    default: {
      NYI("RadixAttnOp not supported for this data type");
    }
  }
}

}  // namespace mllm::cpu
