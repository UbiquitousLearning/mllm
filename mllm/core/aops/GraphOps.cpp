// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/core/aops/GraphOps.hpp"

namespace mllm::aops {

GraphBeginOp::GraphBeginOp(const GraphBeginOpOptions& options) : BaseOp(OpTypes::kGraphBegin), options_(options) {}

void GraphBeginOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void GraphBeginOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void GraphBeginOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void GraphBeginOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void GraphBeginOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

GraphEndOp::GraphEndOp(const GraphEndOpOptions& options) : BaseOp(OpTypes::kGraphEnd), options_(options) {}

void GraphEndOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void GraphEndOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void GraphEndOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void GraphEndOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void GraphEndOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

}  // namespace mllm::aops
