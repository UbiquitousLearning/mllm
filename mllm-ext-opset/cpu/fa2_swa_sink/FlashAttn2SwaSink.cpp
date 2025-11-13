// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm-ext-opset/cpu/fa2_swa_sink/FlashAttn2SwaSink.hpp"
#include "mllm-ext-opset/cpu/fa2_swa_sink/fa2_swa_sink_fwd_bshd.hpp"

namespace mllm::ext_opset::cpu {

void FlashAttention2SwaSink::load(const mllm::ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; };

void FlashAttention2SwaSink::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs,
                                   std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (mllm::ir::IRContext*)trace_context;
  auto i_irs = mllm::ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = mllm::ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<mllm::ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
};

void FlashAttention2SwaSink::forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
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

void FlashAttention2SwaSink::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // CHECK sliding window is set
  MLLM_RT_ASSERT(options_.sliding_window != -1);

  // CHECK inputs
  MLLM_RT_ASSERT(inputs.size() >= 3);
  // CHECK QKV [B, S, H, D]
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  MLLM_RT_ASSERT(q.rank() == 4 && k.rank() == 4 && v.rank() == 4);
  MLLM_RT_ASSERT_EQ(k.size(2), v.size(2));
  MLLM_RT_ASSERT_EQ(q.size(-1), k.size(-1));
  if (options_.s_aux_enable) {
    MLLM_RT_ASSERT_EQ(inputs.size(), 4);
    auto& s_aux = inputs[3];
    MLLM_RT_ASSERT_EQ(s_aux.rank(), 1);
    MLLM_RT_ASSERT_EQ(s_aux.size(0), q.size(2));
  }

  outputs.emplace_back(mllm::Tensor::empty({v.size(0), q.size(1), q.size(2), v.size(3)}, v.dtype(), v.device()));
}

void FlashAttention2SwaSink::setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::ext_opset::cpu

MLLM_PLUGIN_OP_INTERFACE_DEFINE_BEGIN
void* createFlashAttention2SwaSinkFactory() { return new mllm::ext_opset::cpu::FlashAttention2SwaSinkFactory(); };

void freeFlashAttention2SwaSinkFactory(void* factory) {
  delete static_cast<mllm::ext_opset::cpu::FlashAttention2SwaSinkFactory*>(factory);
};

void* opPackageDescriptor() {
  auto package = new PluginOpPackageDescriptor{
      .version = MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_VERSION,
      .name = "MllmExtOpSet.CPU.FlashAttn2SwaSink",
      .device_type = 1,
      .op_factories_count = 1,
      .op_factories_names =
          {
              "flash_attention_2_swa_sink",
          },
      .op_factory_create_funcs =
          {
              createFlashAttention2SwaSinkFactory,
          },
      .op_factory_free_funcs =
          {
              freeFlashAttention2SwaSinkFactory,
          },
  };
  return package;
}
MLLM_PLUGIN_OP_INTERFACE_DEFINE_END
