// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/PadOp.hpp"

namespace mllm::aops {

PadOp::PadOp(const PadOpOptions& options) : BaseOp(OpTypes::kPad), options_(options) {}

void PadOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void PadOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

void PadOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("PadOp::forward is not implemented in the base class");
}

void PadOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

void PadOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

}  // namespace mllm::aops
