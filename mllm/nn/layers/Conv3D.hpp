#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/Conv3DOp.hpp"

namespace mllm::nn {

class Conv3D : public Layer {
 public:
  Conv3D();

  Conv3D(int32_t in_channels, int32_t out_channels, const std::vector<int32_t>& kernel_size,
         const std::vector<int32_t>& stride_size, bool bias = true,
         aops::Conv3DOpImplType impl_type = aops::Conv3DOpImplType::kDefault);

  explicit Conv3D(const aops::Conv3DOpOptions& options);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
