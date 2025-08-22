// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ParamOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

ParamOp::ParamOp(const ParamOpOptions& options) : BaseOp(OpTypes::kParam), options_(options) {}

void ParamOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(options_.name);
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(options_.name);
      break;
    }
    default: NYI("Unsupported model file version")
  }

  // No matter v1, v2, ..., we need to reshape the weight
  if (!options_.shape.empty()) { weight_ = weight_.view(options_.shape); }
}

void ParamOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ParamOp>(shared_from_this(), i_irs, o_irs);
}

void ParamOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void ParamOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void ParamOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

ParameterFile::ptr_t ParamOp::getParams() {
  auto p = ParameterFile::create();
  p->push(options_.name, weight_);
  return p;
}

}  // namespace mllm::aops
