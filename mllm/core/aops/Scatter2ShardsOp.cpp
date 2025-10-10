// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/Scatter2ShardsOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

Scatter2ShardsOp::Scatter2ShardsOp(const Scatter2ShardsOpOptions& options)
    : BaseOp(OpTypes::kScatter2Shards), options_(options) {}

void Scatter2ShardsOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void Scatter2ShardsOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("Scatter2ShardsOp::trace can't be traced in v2.0.0 right now. Pls send us a feature request issues on github if "
            "you need this.");
}

void Scatter2ShardsOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("Scatter2ShardsOp::forward not implemented in aops base.");
}

void Scatter2ShardsOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // scatter op has no output. It only has inputs.
  MLLM_EMPTY_SCOPE;
}

void Scatter2ShardsOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
