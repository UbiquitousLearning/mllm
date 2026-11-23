// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/LinearShadowCorrection.hpp"

namespace mllm::nn {

LinearShadowCorrection::LinearShadowCorrection() : LinearShadowCorrection(0, 0, 16) {}

LinearShadowCorrection::LinearShadowCorrection(int32_t in_channels, int32_t out_channels, int32_t activation_bits)
    : LinearShadowCorrection(aops::LinearShadowCorrectionOpOptions{
          .in_channels = in_channels, .out_channels = out_channels, .activation_bits = activation_bits}) {}

LinearShadowCorrection::LinearShadowCorrection(const aops::LinearShadowCorrectionOpOptions& options)
    : Layer(OpTypes::kLinearShadowCorrection, options) {}

}  // namespace mllm::nn
