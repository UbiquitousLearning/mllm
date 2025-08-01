// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/CopyOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

CopyOp::CopyOp(const CopyOpOptions& options) : BaseOp(OpTypes::kCopy), options_(options) {}

void CopyOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void CopyOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CopyOp>(shared_from_this(), i_irs, o_irs);
}

void CopyOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("CopyOp::forward not implemented in aops base.");
}

void CopyOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Do nothing.
  MLLM_EMPTY_SCOPE;
}

void CopyOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Do nothing.
  MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::aops