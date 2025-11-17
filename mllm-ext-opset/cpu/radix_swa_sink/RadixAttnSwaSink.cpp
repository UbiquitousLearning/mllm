// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/backends/cpu/kernels/common/radix_attn/arch.hpp"
#include "mllm-ext-opset/cpu/radix_swa_sink/RadixAttnSwaSink.hpp"
#include "mllm-ext-opset/cpu/radix_swa_sink/radix_swa_sink_fwd_bshd.hpp"

namespace mllm::ext_opset::cpu {

void RadixAttnSwaSink::load(const mllm::ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; };

void RadixAttnSwaSink::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (mllm::ir::IRContext*)trace_context;
  auto i_irs = mllm::ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = mllm::ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<mllm::ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
};

void RadixAttnSwaSink::forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
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
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH)
      switch (options_.pattern) {
        case RadixAttnSwaSinkPattern::kDecode: {
          fwd_bshd_decode<::mllm::cpu::radix_attn::details::__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                          mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V,
                                       options_.sliding_window, options_.cur_seq_len, Q.ptr<mllm_fp32_t>(),
                                       K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), S_AUX.ptr<mllm_fp32_t>(),
                                       O.ptr<mllm_fp32_t>(), options_.getThreads());
          break;
        }
        case RadixAttnSwaSinkPattern::kPrefill: {
          fwd_bshd_prefill<::mllm::cpu::radix_attn::details::__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                           mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V,
                                        options_.sliding_window, options_.cur_seq_len, Q.ptr<mllm_fp32_t>(),
                                        K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), S_AUX.ptr<mllm_fp32_t>(),
                                        O.ptr<mllm_fp32_t>(), options_.getThreads());
          break;
        }
        case RadixAttnSwaSinkPattern::kAppend: {
          // TODO
          break;
        }
      }
#elif defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)
      switch (options_.pattern) {
        case RadixAttnSwaSinkPattern::kDecode: {
          fwd_bshd_decode<::mllm::cpu::radix_attn::details::__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                          mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V,
                                       options_.sliding_window, options_.cur_seq_len, Q.ptr<mllm_fp32_t>(),
                                       K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), S_AUX.ptr<mllm_fp32_t>(),
                                       O.ptr<mllm_fp32_t>(), options_.getThreads());
          break;
        }
        case RadixAttnSwaSinkPattern::kPrefill: {
          fwd_bshd_prefill<::mllm::cpu::radix_attn::details::__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                           mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V,
                                        options_.sliding_window, options_.cur_seq_len, Q.ptr<mllm_fp32_t>(),
                                        K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), S_AUX.ptr<mllm_fp32_t>(),
                                        O.ptr<mllm_fp32_t>(), options_.getThreads());
          break;
        }
        case RadixAttnSwaSinkPattern::kAppend: {
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

void RadixAttnSwaSink::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // CHECK sliding window is set
  MLLM_RT_ASSERT(options_.sliding_window != -1);

  // CHECK inputs
  MLLM_RT_ASSERT(inputs.size() >= 3);
  // CHECK QKV [B, S, H, D]
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  MLLM_RT_ASSERT(q.rank() == 4 && k.rank() == 1 && v.rank() == 1);
  if (options_.s_aux_enable) {
    MLLM_RT_ASSERT_EQ(inputs.size(), 4);
    auto& s_aux = inputs[3];
    MLLM_RT_ASSERT_EQ(s_aux.rank(), 1);
    MLLM_RT_ASSERT_EQ(s_aux.size(0), q.size(2));
  }

  outputs.emplace_back(mllm::Tensor::empty({options_.B, q.size(1), options_.q_head, options_.D_V}, q.dtype(), q.device()));
}

void RadixAttnSwaSink::setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::ext_opset::cpu

MLLM_PLUGIN_OP_INTERFACE_DEFINE_BEGIN
void* createRadixAttnSwaSinkFactory() { return new mllm::ext_opset::cpu::RadixAttnSwaSinkFactory(); };

void freeRadixAttnSwaSinkFactory(void* factory) {
  delete static_cast<mllm::ext_opset::cpu::RadixAttnSwaSinkFactory*>(factory);
};

void* opPackageDescriptor() {
  auto package = new PluginOpPackageDescriptor{
      .version = MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_VERSION,
      .name = "MllmExtOpSet.CPU.RadixAttnSwaSink",
      .device_type = 1,
      .op_factories_count = 1,
      .op_factories_names =
          {
              "radix_attn_swa_sink",
          },
      .op_factory_create_funcs =
          {
              createRadixAttnSwaSinkFactory,
          },
      .op_factory_free_funcs =
          {
              freeRadixAttnSwaSinkFactory,
          },
  };
  return package;
}
MLLM_PLUGIN_OP_INTERFACE_DEFINE_END
