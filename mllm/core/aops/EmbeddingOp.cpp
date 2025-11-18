// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/EmbeddingOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"

namespace mllm::aops {

EmbeddingOp::EmbeddingOp(const EmbeddingOpOptions& options) : BaseOp(OpTypes::kEmbedding), options_(options) {}

void EmbeddingOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      weight_ = weight_.view({options_.vocab_size, options_.hidden_size});
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      break;
    }
    default: NYI("Unsupported model file version")
  }
}

void EmbeddingOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;

  // Register Params
  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    auto init_node = ir_ctx->lookupSymbolTable("init");
    if (!init_node) {
      MLLM_ERROR("EmbeddingOp::trace: 'init' symbol not found in symbol table");
      return;
    }

    std::shared_ptr<ir::Region> init_region = nullptr;
    if (init_node->isa_<ir::graph::SubGraphOp>()) {
      auto subgraph_op = init_node->cast_<ir::graph::SubGraphOp>();
      init_region = subgraph_op->getTopRegion();
    } else {
      MLLM_ERROR("EmbeddingOp::trace: 'init' is not SubGraphOp");
      return;
    }

    ir::IRWriterGuard guard(ir_ctx, init_region);
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
  }

  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::EmbeddingOp>(shared_from_this(), i_irs, o_irs);
}

void EmbeddingOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("EmbeddingOp::forward not implemented in aops base.");
}

void EmbeddingOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto shape = i.shape();
  std::vector<int32_t> o_shape{/*batch*/ shape[0], /*seq*/ shape[1],
                               /*feat dim*/ options_.hidden_size};

  // FIXME: We should tell embedding output to use what kinds of data types. Currently it's hardcoded to float32.
  outputs.emplace_back(Tensor::empty(o_shape, kFloat32, i.device()));
}

void EmbeddingOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ParameterFile::ptr_t EmbeddingOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  return p;
}
}  // namespace mllm::aops
