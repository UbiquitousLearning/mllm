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
  // TODO
  MLLM_EMPTY_SCOPE;
}

void PagedAttnOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
  MLLM_EMPTY_SCOPE;
}

ParameterFile::ptr_t PagedAttnOp::getParams() {
  auto p = ParameterFile::create();
  return p;
}

}  // namespace mllm::aops
