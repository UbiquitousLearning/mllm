// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/PagedAttnOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

PagedAttnOp::PagedAttnOp(const PagedAttnOpOptions& options) : BaseOp(OpTypes::kPagedAttn), options_(options) {}

void PagedAttnOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void PagedAttnOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::PagedAttnOp>(shared_from_this(), i_irs, o_irs);
}

void PagedAttnOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void PagedAttnOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Inputs should in BSHD format.
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& index = inputs[3];

  MLLM_RT_ASSERT_EQ(index.rank(), 1);
  MLLM_RT_ASSERT_EQ(index.dtype(), kInt32);
  auto q_shape = q.shape();
  auto B = q_shape[0];
  auto S_Q = q_shape[1];
  auto H = q_shape[2];
  auto D = q_shape[3];

  auto f_shape = Tensor::shape_t{B, S_Q, H, D};

  auto out_dtype = kFloat32;
  switch (options_.impl_type) {
    case PagedAttnImplType::kAllFp32: out_dtype = kFloat32; break;
    case PagedAttnImplType::kDefault: out_dtype = q.dtype(); break;
  }

  outputs.emplace_back(Tensor::empty(f_shape, out_dtype, q.device()));

  if (options_.need_attn_weights) {
    auto acc_dtype = out_dtype;
    switch (options_.impl_type) {
      case PagedAttnImplType::kAllFp32: acc_dtype = kFloat32; break;
      case PagedAttnImplType::kDefault: acc_dtype = q.dtype(); break;
    }
    auto S_V = index.shape()[0];
    // Push attn weights
    outputs.emplace_back(Tensor::empty({B, H, S_Q, S_V}, acc_dtype, q.device()));
  } else {
    outputs.emplace_back(Tensor::nil());
  }
}

void PagedAttnOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ParameterFile::ptr_t PagedAttnOp::getParams() {
  auto p = ParameterFile::create();
  return p;
}

}  // namespace mllm::aops
