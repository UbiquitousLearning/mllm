// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/ConvTranspose1DOp.hpp"

namespace mllm::nn {

class ConvTranspose1D : public Layer {
 public:
  ConvTranspose1D();

  ConvTranspose1D(int32_t in_channels, int32_t out_channels, int32_t kernel_size, int32_t stride_size = 1,
                  int32_t padding = 0, int32_t output_padding = 0, int32_t dilation = 1, int32_t groups = 1,
                  bool bias = true);

  explicit ConvTranspose1D(const aops::ConvTranspose1DOpOptions& options);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
