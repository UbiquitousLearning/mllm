// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/FillOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

FillOp::FillOp(const FillOpOptions& options) : BaseOp(OpTypes::kFill), options_(options) {}

void FillOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void FillOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::FillOp>(shared_from_this(), i_irs, o_irs);
}

void FillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("FillOp::forward not implemented in aops base.");
}

void FillOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { outputs.emplace_back(inputs[0]); }

void FillOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // The fillop is performed inplace
  // There is no need to alloc output again!
  MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::aops