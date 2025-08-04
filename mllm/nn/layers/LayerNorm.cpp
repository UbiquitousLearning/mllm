// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/LayerNormOp.hpp"
#include "mllm/nn/layers/LayerNorm.hpp"

namespace mllm::nn {

LayerNorm::LayerNorm() : Layer(OpTypes::kLayerNorm, aops::LayerNormOpOptions{}) {}

LayerNorm::LayerNorm(const aops::LayerNormOpOptions& options) : Layer(OpTypes::kLayerNorm, options) {}

LayerNorm::LayerNorm(const std::vector<int32_t>& normalized_shape, bool elementwise_affine, bool bias, float eps)
    : Layer(OpTypes::kLayerNorm,
            aops::LayerNormOpOptions{
                .normalized_shape = normalized_shape, .elementwise_affine = elementwise_affine, .bias = bias, .eps = eps}) {}

}  // namespace mllm::nn
