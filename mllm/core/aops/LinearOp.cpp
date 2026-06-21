// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

LinearOp::LinearOp(const LinearOpOptions& options) : BaseOp(OpTypes::kLinear), options_(options) {}

void LinearOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      switch (options_.impl_type) {
        case aops::LinearImplTypes::kDefault: {
          weight_ = weight_.view({options_.out_channels, options_.in_channels});
          if (options_.bias) {
            bias_ = ploader->pull(getName() + ".bias");
            bias_ = bias_.view({options_.out_channels});
          }
          break;
        }
        default: {
          // No need to view.
          MLLM_EMPTY_SCOPE
          break;
        }
      }
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      switch (options_.impl_type) {
        case aops::LinearImplTypes::kBLAS:
        case aops::LinearImplTypes::kDefault: {
          if (options_.bias) {
            bias_ = ploader->pull(getName() + ".bias");
            bias_ = bias_.view({options_.out_channels});
          }
          break;
        }
        default: {
          // No need to view.
          MLLM_EMPTY_SCOPE
          break;
        }
      }
      break;
    }
    default: NYI("Unsupported model file version")
  }
}

void LinearOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;

  // Register Params
  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    ir::IRWriterGuard guard(ir_ctx, ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion());
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
    if (options_.bias) {
      switch (options_.impl_type) {
        case aops::LinearImplTypes::kBLAS:
        case aops::LinearImplTypes::kDefault: {
          ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(bias_));
        }
        default: {
          // No need to view.
          MLLM_EMPTY_SCOPE
          break;
        }
      }
    }
  }

  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::LinearOp>(shared_from_this(), i_irs, o_irs);
}

void LinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LinearOp::forward not implemented in aops base.");
}

void LinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isRedirect()) {
    outputs.emplace_back(inputs[1]);
    return;
  }
  auto& input = inputs[0];
  auto out_shape = input.shape();
  MLLM_RT_ASSERT_EQ(out_shape[out_shape.size() - 1], options_.in_channels);
  out_shape[out_shape.size() - 1] = options_.out_channels;

  outputs.emplace_back(Tensor::empty(out_shape, input.dtype(), input.device()));
}

void LinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isRedirect()) { return; }
  BaseOp::setup(inputs, outputs);
}

ParameterFile::ptr_t LinearOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  if (options_.bias) { p->push(getName() + ".bias", bias_); }
  return p;
}

}  // namespace mllm::aops
