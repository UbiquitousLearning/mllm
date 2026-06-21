#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/Conv2DOp.hpp"

namespace mllm::nn {

class Conv2D : public Layer {
 public:
  Conv2D();

  Conv2D(int32_t in_channels, int32_t out_channels, const std::vector<int32_t>& kernel_size,
         const std::vector<int32_t>& stride_size, const std::vector<int32_t>& padding_size,
         const std::vector<int32_t>& dilation_size, bool bias = true,
         aops::Conv2DOpImplType impl_type = aops::Conv2DOpImplType::kDefault);

  explicit Conv2D(const aops::Conv2DOpOptions& options);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
  MLLM_LAYER_ENABLE_REDIRECT_ATTRIBUTE(Conv2D)
};

}  // namespace mllm::nn
