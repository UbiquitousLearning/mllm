// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/MultimodalRoPEOp.hpp"
#include "mllm/nn/layers/MultimodalRoPE.hpp"

namespace mllm::nn {

MultimodalRoPE::MultimodalRoPE() : Layer(OpTypes::kMultimodalRoPE, aops::MultimodalRoPEOpOptions{}) {}

MultimodalRoPE::MultimodalRoPE(const aops::MultimodalRoPEOpOptions& options) : Layer(OpTypes::kMultimodalRoPE, options) {}

MultimodalRoPE::MultimodalRoPE(const aops::Qwen2VLMultimodalRoPEOpOptions& options,
                               aops::MultimodalRoPEOpOptionsInputType input_type)
    : Layer(OpTypes::kMultimodalRoPE,
            aops::MultimodalRoPEOpOptions{aops::MultimodalRoPEOpOptionsType::kQwen2VL, options, input_type}) {}

}  // namespace mllm::nn
