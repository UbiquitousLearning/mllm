// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/RadixAttnOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

RadixAttnOp::RadixAttnOp(const RadixAttnOpOptions& options) : BaseOp(OpTypes::kRadixAttn), options_(options) {}

void RadixAttnOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void RadixAttnOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("RadixAttnOp::trace is not supported.");
}

void RadixAttnOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("RadixAttnOp::forward not implemented in aops base.");
}

void RadixAttnOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& query = inputs[0];
  auto& key = inputs[1];
  auto& value = inputs[2];

  outputs.emplace_back(Tensor::empty(query.shape(), query.dtype(), query.device()));
}

void RadixAttnOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
