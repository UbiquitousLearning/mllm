// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/VisionRoPEOp.hpp"
#include "mllm/nn/layers/VisionRoPE.hpp"

namespace mllm::nn {

VisionRoPE::VisionRoPE() : Layer(OpTypes::kVisionRoPE, aops::VisionRoPEOpOptions{}) {}

VisionRoPE::VisionRoPE(const aops::VisionRoPEOpOptions& options) : Layer(OpTypes::kVisionRoPE, options) {}

VisionRoPE::VisionRoPE(const aops::VisionRoPEOpOptionsType type, const aops::Qwen2VLRoPEOpOptions& Options)
    : Layer(OpTypes::kVisionRoPE, aops::VisionRoPEOpOptions{.type = type, Options}) {}

}  // namespace mllm::nn
