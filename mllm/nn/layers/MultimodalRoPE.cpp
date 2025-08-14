// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/MultimodalRoPEOp.hpp"
#include "mllm/nn/layers/MultimodalRoPE.hpp"

namespace mllm::nn {

MultimodalRoPE::MultimodalRoPE() : Layer(OpTypes::kMultimodalRoPE, aops::MultimodalRoPEOpOptions{}) {}

MultimodalRoPE::MultimodalRoPE(const aops::MultimodalRoPEOpOptions& options) : Layer(OpTypes::kMultimodalRoPE, options) {}

MultimodalRoPE::MultimodalRoPE(const aops::Qwen2VLMultimodalRoPEOpOptions& options)
    : Layer(OpTypes::kMultimodalRoPE, aops::MultimodalRoPEOpOptions{aops::MultimodalRoPEOpOptionsType::kQwen2VL, options}) {}

}  // namespace mllm::nn
