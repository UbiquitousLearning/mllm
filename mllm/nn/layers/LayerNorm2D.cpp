// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/LayerNorm2DOp.hpp"
#include "mllm/nn/layers/LayerNorm2D.hpp"

namespace mllm::nn {

LayerNorm2D::LayerNorm2D() : Layer(OpTypes::kLayerNorm2D, aops::LayerNorm2DOpOptions{}) {}

LayerNorm2D::LayerNorm2D(const aops::LayerNorm2DOpOptions& options) : Layer(OpTypes::kLayerNorm2D, options) {}

LayerNorm2D::LayerNorm2D(const int32_t num_channels, float eps)
    : Layer(OpTypes::kLayerNorm2D, aops::LayerNorm2DOpOptions{.num_channels = num_channels, .eps = eps}) {}

}  // namespace mllm::nn
