#pragma once

#include <cstdint>
#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/AvgPool1dOp.hpp"

namespace mllm::nn {

class AvgPool1d : public Layer {
 public:
  AvgPool1d();

  AvgPool1d(int32_t kernel_size, int32_t stride, int32_t padding = 0, bool ceil_mode = false,
            bool count_include_pad = true);

  explicit AvgPool1d(const aops::AvgPool1dOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
