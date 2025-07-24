/**
 * @file Linear.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-24
 *
 */
#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::nn {

class Linear : public Layer {
 public:
  Linear();

  Linear(int32_t in_channels, int32_t out_channels, bool bias = true,
         aops::LinearImplTypes impl_type = aops::LinearImplTypes::kDefault);

  explicit Linear(const aops::LinearOptions& options);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;
};

}  // namespace mllm::nn
