// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/backends/cpu/kernels/common/radix_attn/arch.hpp"
#include "mllm-ext-opset/cpu/radix_attn_relax/RadixAttnRelax.hpp"
#include "mllm-ext-opset/cpu/radix_attn_relax/radix_attn_relax_fwd_bshd.hpp"

namespace mllm::ext_opset::cpu {

void RadixAttnRelax::load(const mllm::ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; };

void RadixAttnRelax::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (mllm::ir::IRContext*)trace_context;
  auto i_irs = mllm::ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = mllm::ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<mllm::ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
};

void RadixAttnRelax::forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
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
      fwd_bshd<::mllm::cpu::radix_attn::details::__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t>(
          B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V, Q.ptr<mllm_fp32_t>(),
          K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), O.ptr<mllm_fp32_t>(), options_.getThreads());
#elif defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)
      fwd_bshd<::mllm::cpu::radix_attn::details::__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t>(
          B, options_.q_head, options_.kv_head, S_Q, S_KV, options_.D_QK, options_.D_V, Q.ptr<mllm_fp32_t>(),
          K.ptr<mllm_fp32_t*>(), V.ptr<mllm_fp32_t*>(), O.ptr<mllm_fp32_t>(), options_.getThreads());
#endif
      break;
    }
    default: NYI("RadixAttnRelax::forward not support dtype {}", nameOfType(Q.dtype())); break;
  }
}

void RadixAttnRelax::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // CHECK inputs
  MLLM_RT_ASSERT(inputs.size() >= 3);
  // CHECK QKV [B, S, H, D]
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  MLLM_RT_ASSERT(q.rank() == 4 && k.rank() == 1 && v.rank() == 1);
  outputs.emplace_back(mllm::Tensor::empty({options_.B, q.size(1), options_.q_head, options_.D_V}, q.dtype(), q.device()));
}

void RadixAttnRelax::setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::ext_opset::cpu

MLLM_PLUGIN_OP_INTERFACE_DEFINE_BEGIN
void* createRadixAttnRelaxFactory() { return new mllm::ext_opset::cpu::RadixAttnRelaxFactory(); };

void freeRadixAttnRelaxFactory(void* factory) { delete static_cast<mllm::ext_opset::cpu::RadixAttnRelaxFactory*>(factory); };

void* opPackageDescriptor() {
  auto package = new PluginOpPackageDescriptor{
      .version = MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_VERSION,
      .name = "MllmExtOpSet.CPU.RadixAttnRelax",
      .device_type = 1,
      .op_factories_count = 1,
      .op_factories_names =
          {
              "radix_attn_relax",
          },
      .op_factory_create_funcs =
          {
              createRadixAttnRelaxFactory,
          },
      .op_factory_free_funcs =
          {
              freeRadixAttnRelaxFactory,
          },
  };
  return package;
}
MLLM_PLUGIN_OP_INTERFACE_DEFINE_END
